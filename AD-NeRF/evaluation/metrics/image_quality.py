"""
图像质量指标: PSNR, SSIM 和 FID

这些指标用于评估生成图像相对于真值图像的视觉质量
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Union
from pathlib import Path
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_psnr(gen_images: Union[List[str], np.ndarray], 
                 gt_images: Union[List[str], np.ndarray],
                 data_range: float = 1.0) -> Tuple[float, List[float]]:
    """
    计算生成图像和真值图像之间的峰值信噪比 (PSNR)
    
    PSNR 越高表示图像质量越好（通常好的结果在 20-40 dB）
    
    参数:
        gen_images: 生成图像路径列表或图像数组
        gt_images: 真值图像路径列表或图像数组
        data_range: 图像数据范围 (1.0 表示 [0,1], 255 表示 [0,255])
    
    返回:
        (平均PSNR, 每帧PSNR列表) 的元组
    """
    # 如果提供的是路径，则加载图像
    if isinstance(gen_images, list) and isinstance(gen_images[0], str):
        gen_images = load_images(gen_images, normalize=True)
        gt_images = load_images(gt_images, normalize=True)
        data_range = 1.0
    
    psnr_scores = []
    
    for gen_img, gt_img in zip(gen_images, gt_images):
        # 确保图像为浮点格式
        if gen_img.dtype != np.float32 and gen_img.dtype != np.float64:
            gen_img = gen_img.astype(np.float32) / 255.0
            gt_img = gt_img.astype(np.float32) / 255.0
            data_range = 1.0
        
        score = psnr(gt_img, gen_img, data_range=data_range)
        psnr_scores.append(score)
    
    mean_psnr = np.mean(psnr_scores)
    
    return mean_psnr, psnr_scores


def compute_ssim(gen_images: Union[List[str], np.ndarray],
                 gt_images: Union[List[str], np.ndarray],
                 data_range: float = 1.0,
                 multichannel: bool = True) -> Tuple[float, List[float]]:
    """
    计算生成图像和真值图像之间的结构相似性指数 (SSIM)
    
    SSIM 值范围从 -1 到 1，其中 1 表示完全相似
    
    参数:
        gen_images: 生成图像路径列表或图像数组
        gt_images: 真值图像路径列表或图像数组
        data_range: 图像数据范围 (1.0 表示 [0,1], 255 表示 [0,255])
        multichannel: 图像是否有多个通道 (RGB)
    
    返回:
        (平均SSIM, 每帧SSIM列表) 的元组
    """
    # 如果提供的是路径，则加载图像
    if isinstance(gen_images, list) and isinstance(gen_images[0], str):
        gen_images = load_images(gen_images, normalize=True)
        gt_images = load_images(gt_images, normalize=True)
        data_range = 1.0
    
    ssim_scores = []
    
    for gen_img, gt_img in zip(gen_images, gt_images):
        # 确保图像为浮点格式
        if gen_img.dtype != np.float32 and gen_img.dtype != np.float64:
            gen_img = gen_img.astype(np.float32) / 255.0
            gt_img = gt_img.astype(np.float32) / 255.0
            data_range = 1.0
        
        # 计算 SSIM (channel_axis 用于 scikit-image >= 0.19)
        try:
            score = ssim(gt_img, gen_img, data_range=data_range, channel_axis=2)
        except TypeError:
            # 兼容旧版本 scikit-image
            score = ssim(gt_img, gen_img, data_range=data_range, multichannel=multichannel)
        
        ssim_scores.append(score)

    mean_ssim = np.mean(ssim_scores)
    return mean_ssim, ssim_scores


def compute_fid(gen_dir: str,
                gt_dir: str,
                batch_size: int = 50,
                device: str = 'cuda',
                dims: int = 2048,
                num_workers: int = 4) -> float:
    """
    计算生成图像和真值图像之间的 Fréchet Inception Distance (FID)
    
    FID 衡量特征分布之间的距离。越低越好。
    典型值: <10 为高质量合成, <50 为可接受质量
    
    参数:
        gen_dir: 包含生成图像的目录
        gt_dir: 包含真值图像的目录
        batch_size: 处理批次大小
        device: 使用的设备 ('cuda' 或 'cpu')
        dims: Inception 特征的维度 (2048 是标准值)
        num_workers: 数据加载工作线程数
    
    返回:
        FID 分数 (float)
    """
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        raise ImportError(
            "未安装 pytorch-fid。使用以下命令安装: pip install pytorch-fid"
        )
    
    # 检查目录是否存在
    if not os.path.exists(gen_dir):
        raise ValueError(f"未找到生成图像目录: {gen_dir}")
    if not os.path.exists(gt_dir):
        raise ValueError(f"未找到真值图像目录: {gt_dir}")
    
    # 如果请求 CUDA 但不可用，则检查
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU 计算 FID")
        device = 'cpu'
    
    print(f"使用 {device} 计算 FID...")
    
    # 计算 FID
    fid_value = fid_score.calculate_fid_given_paths(
        [gen_dir, gt_dir],
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=num_workers
    )
    
    return fid_value


def load_images(image_paths: List[str], normalize: bool = True) -> np.ndarray:
    """
    从文件路径加载图像
    
    参数:
        image_paths: 图像文件路径列表
        normalize: 是否归一化到 [0, 1]
    
    返回:
        图像的 Numpy 数组 (N, H, W, 3)
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"加载图像失败: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if normalize:
            img = img.astype(np.float32) / 255.0
        images.append(img)
    
    return np.array(images)


def compute_all_image_metrics(gen_dir: str, 
                              gt_dir: str,
                              device: str = 'cuda') -> dict:
    """
    一次性计算所有图像质量指标 (PSNR, SSIM, FID)
    
    参数:
        gen_dir: 包含生成图像的目录
        gt_dir: 包含真值图像的目录
        device: FID 计算使用的设备
    
    返回:
        包含所有指标的字典
    """
    # 获取配对的图像路径
    gen_images = sorted(Path(gen_dir).glob('*.jpg'))
    gt_images = sorted(Path(gt_dir).glob('*.jpg'))
    
    if len(gen_images) == 0:
        raise ValueError(f"在 {gen_dir} 中未找到图像")
    if len(gt_images) == 0:
        raise ValueError(f"在 {gt_dir} 中未找到图像")
    
    # 确保图像数量相等
    min_len = min(len(gen_images), len(gt_images))
    gen_images = [str(p) for p in gen_images[:min_len]]
    gt_images = [str(p) for p in gt_images[:min_len]]
    
    print(f"正在评估 {len(gen_images)} 对图像...")
    
    # 计算 PSNR
    print("计算 PSNR...")
    mean_psnr, psnr_list = compute_psnr(gen_images, gt_images)
    
    # 计算 SSIM
    print("计算 SSIM...")
    mean_ssim, ssim_list = compute_ssim(gen_images, gt_images)
    
    # 计算 FID
    print("计算 FID...")
    try:
        fid_value = compute_fid(gen_dir, gt_dir, device=device)
    except Exception as e:
        print(f"警告: FID 计算失败: {e}")
        fid_value = None
    
    results = {
        'psnr': {
            'mean': mean_psnr,
            'std': np.std(psnr_list),
            'per_frame': psnr_list
        },
        'ssim': {
            'mean': mean_ssim,
            'std': np.std(ssim_list),
            'per_frame': ssim_list
        },
        'fid': fid_value,
        'num_frames': len(gen_images)
    }
    
    return results


if __name__ == '__main__':
    # 测试指标
    import argparse
    
    parser = argparse.ArgumentParser(description='测试图像质量指标')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='生成图像目录')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='真值图像目录')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    results = compute_all_image_metrics(args.gen_dir, args.gt_dir, args.device)
    
    print("\n" + "="*50)
    print("图像质量指标结果")
    print("="*50)
    print(f"PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
    print(f"SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
    if results['fid'] is not None:
        print(f"FID:  {results['fid']:.2f}")
    print(f"帧数: {results['num_frames']}")
    print("="*50)
