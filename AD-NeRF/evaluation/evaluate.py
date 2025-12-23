"""
AD-NeRF 说话人头部合成主评估脚本

计算以下综合指标:
- 图像质量: PSNR, SSIM, FID
- 面部特征点: NIOE

使用方法:
    python evaluate.py --subject Obama --metrics psnr ssim fid nioe
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

# 将父目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

from metrics.image_quality import compute_all_image_metrics
from metrics.landmark_metrics import compute_landmark_metrics
from utils.extract_gt_frames import extract_gt_from_ori_imgs
from utils.visualization import plot_metrics, create_report_table


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='评估 AD-NeRF 生成的视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估 Obama 的所有指标
  python evaluate.py --subject Obama --metrics all
  
  # 仅评估图像质量指标
  python evaluate.py --subject Obama --metrics psnr ssim fid
  
  # 使用自定义路径评估
  python evaluate.py --subject Obama --gen_dir custom/path --gt_dir custom/gt
  
  # 跳过真值提取（如果已完成）
  python evaluate.py --subject Obama --skip_gt_extraction
        """
    )
    
    # 必需参数
    parser.add_argument('--subject', type=str, required=True,
                        help='主题名称 (例如: Obama, Jae-in, Lieu, Macron)')
    
    # 指标选择
    parser.add_argument('--metrics', nargs='+', 
                        default=['psnr', 'ssim', 'fid'],
                        choices=['all', 'psnr', 'ssim', 'fid', 'nioe'],
                        help='要计算的指标 (默认: psnr ssim fid)')
    
    # 目录路径
    parser.add_argument('--dataset_dir', type=str, 
                        default='AD-NeRF/dataset',
                        help='数据集目录路径 (默认: AD-NeRF/dataset)')
    
    parser.add_argument('--gen_dir', type=str, default=None,
                        help='生成图像目录 (默认: 从日志自动检测)')
    
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='真值图像目录 (默认: {subject}/gt_frames)')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录 (默认: AD-NeRF/evaluation/results/{subject})')
    
    # 真值提取
    parser.add_argument('--skip_gt_extraction', action='store_true',
                        help='跳过真值提取（假设真值帧已存在）')
    
    parser.add_argument('--testskip', type=int, default=1,
                        help='真值提取的帧跳过因子 (默认: 1)')
    
    # 设备和性能
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='计算设备 (默认: cuda)')
    
    parser.add_argument('--batch_size', type=int, default=50,
                        help='FID 计算的批次大小 (默认: 50)')
    
    # 可视化
    parser.add_argument('--no_visualization', action='store_true',
                        help='跳过可视化生成')
    
    parser.add_argument('--save_json', action='store_true',
                        help='将结果保存为 JSON 文件')
    
    return parser.parse_args()


def setup_paths(args) -> Dict[str, str]:
    """
    设置并验证所有路径
    
    返回:
        包含所有路径的字典
    """
    paths = {}
    
    # 数据集目录
    subject_dir = Path(args.dataset_dir) / args.subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"未找到主题目录: {subject_dir}")
    
    paths['subject_dir'] = str(subject_dir)
    
    # 生成图像目录
    if args.gen_dir:
        paths['gen_dir'] = args.gen_dir
    else:
        # 从日志自动检测
        gen_dir = subject_dir / 'logs' / f'{args.subject}_com' / 'test_aud_rst'
        if not gen_dir.exists():
            raise FileNotFoundError(
                f"在 {gen_dir} 未找到生成的图像。"
                "请使用 --gen_dir 指定路径"
            )
        paths['gen_dir'] = str(gen_dir)
    
    # 真值图像目录
    if args.gt_dir:
        paths['gt_dir'] = args.gt_dir
    else:
        paths['gt_dir'] = str(subject_dir / 'gt_frames')
    
    # 输出目录
    if args.output_dir:
        paths['output_dir'] = args.output_dir
    else:
        paths['output_dir'] = str(Path('AD-NeRF/evaluation/results') / args.subject)
    
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    # Transforms 文件
    paths['transforms_json'] = str(subject_dir / 'transforms_val.json')
    
    # 原始图像（用于真值提取）
    paths['ori_imgs_dir'] = str(subject_dir / 'ori_imgs')
    
    return paths


def extract_ground_truth(paths: Dict, args) -> None:
    """如需要，提取真值帧"""
    if args.skip_gt_extraction:
        print("跳过真值提取...")
        if not os.path.exists(paths['gt_dir']):
            raise FileNotFoundError(
                f"未找到真值目录: {paths['gt_dir']}。"
                "不使用 --skip_gt_extraction 运行以提取真值帧。"
            )
        return
    
    if os.path.exists(paths['gt_dir']):
        print(f"真值目录已存在: {paths['gt_dir']}")
        response = input("重新提取真值帧? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n" + "="*60)
    print("提取真值帧")
    print("="*60)
    
    extract_gt_from_ori_imgs(
        ori_imgs_dir=paths['ori_imgs_dir'],
        transforms_json=paths['transforms_json'],
        output_dir=paths['gt_dir'],
        testskip=args.testskip
    )
    
    print("真值提取完成!")


def evaluate_metrics(paths: Dict, args) -> Dict:
    """
    运行选定指标的评估
    
    返回:
        包含所有结果的字典
    """
    results = {
        'subject': args.subject,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 确定要计算的指标
    metrics_to_compute = args.metrics
    if 'all' in metrics_to_compute:
        metrics_to_compute = ['psnr', 'ssim', 'fid', 'nioe']
    
    print("\n" + "="*60)
    print(f"正在评估: {', '.join(metrics_to_compute).upper()}")
    print("="*60)
    
    # 图像质量指标
    if any(m in metrics_to_compute for m in ['psnr', 'ssim', 'fid']):
        print("\n📊 计算图像质量指标...")
        try:
            img_results = compute_all_image_metrics(
                gen_dir=paths['gen_dir'],
                gt_dir=paths['gt_dir'],
                device=args.device
            )
            results.update(img_results)
            
            print(f"  ✓ PSNR: {img_results['psnr']['mean']:.2f} dB")
            print(f"  ✓ SSIM: {img_results['ssim']['mean']:.4f}")
            if img_results['fid'] is not None:
                print(f"  ✓ FID:  {img_results['fid']:.2f}")
        except Exception as e:
            print(f"  ✗ 计算图像质量指标时出错: {e}")
    
    # 面部特征点指标
    if 'nioe' in metrics_to_compute:
        print("\n📍 计算面部特征点指标...")
        try:
            landmark_results = compute_landmark_metrics(
                gen_dir=paths['gen_dir'],
                gt_dir=paths['gt_dir'],
                device=args.device,
                subset='mouth'  # 专注于说话人头部的嘴部区域
            )
            results['nioe'] = landmark_results['nioe']
            
            print(f"  ✓ NIOE: {landmark_results['nioe']['mean']:.4f}")
        except Exception as e:
            print(f"  ✗ 计算面部特征点指标时出错: {e}")
            print(f"     请确保已安装 face-alignment: pip install face-alignment")
    
    return results


def save_results(results: Dict, paths: Dict, args) -> None:
    """保存评估结果"""
    output_dir = Path(paths['output_dir'])
    
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    
    # 保存文本报告
    report_path = output_dir / 'evaluation_report.txt'
    create_report_table(results, str(report_path))
    
    # 保存 JSON
    if args.save_json:
        json_path = output_dir / 'evaluation_results.json'
        # 移除不可序列化的项
        json_results = {k: v for k, v in results.items() 
                       if k not in ['video_path']}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"已保存 JSON 结果到 {json_path}")
    
    # 创建可视化
    if not args.no_visualization and ('psnr' in results or 'ssim' in results):
        print("创建可视化...")
        plot_path = output_dir / 'metrics_plot.png'
        plot_metrics(results, str(plot_path), title=f'{args.subject} 评估结果')
    
    print(f"\n所有结果已保存到: {output_dir}")


def main():
    """主评估流程"""
    args = parse_args()
    
    print("="*60)
    print("AD-NeRF 评估流程")
    print("="*60)
    print(f"主题: {args.subject}")
    print(f"指标: {', '.join(args.metrics)}")
    print(f"设备: {args.device}")
    
    # 设置路径
    try:
        paths = setup_paths(args)
        print(f"\n生成的图像: {paths['gen_dir']}")
        print(f"真值: {paths['gt_dir']}")
        print(f"输出: {paths['output_dir']}")
    except Exception as e:
        print(f"\n✗ 设置路径时出错: {e}")
        return 1
    
    # 提取真值帧
    try:
        extract_ground_truth(paths, args)
    except Exception as e:
        print(f"\n✗ 提取真值时出错: {e}")
        return 1
    
    # 运行评估
    try:
        results = evaluate_metrics(paths, args)
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 保存结果
    try:
        save_results(results, paths, args)
    except Exception as e:
        print(f"\n✗ 保存结果时出错: {e}")
        return 1
    
    print("\n" + "="*60)
    print("✓ 评估完成!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
