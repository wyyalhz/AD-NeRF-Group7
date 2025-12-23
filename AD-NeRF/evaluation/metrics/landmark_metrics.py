"""
基于特征点的指标: NIOE (归一化眼间距误差)

该指标衡量生成图像中面部特征点位置的准确性
"""

import numpy as np
import torch
from typing import List, Tuple, Union, Optional
import cv2
from pathlib import Path


def compute_nioe(gen_images: Union[List[str], np.ndarray],
                 gt_images: Union[List[str], np.ndarray],
                 detector: Optional[object] = None,
                 device: str = 'cuda',
                 landmarks_subset: Optional[List[int]] = None) -> Tuple[float, List[float]]:
    """
    计算面部特征点的归一化眼间距误差 (NIOE)
    
    NIOE 通过眼间距（双眼之间的距离）归一化特征点误差，
    使指标具有尺度不变性。值越低表示特征点准确性越高。
    
    参数:
        gen_images: 生成图像路径列表或 numpy 数组
        gt_images: 真值图像路径列表或 numpy 数组
        detector: 面部对齐检测器（如果为 None，将创建一个）
        device: 特征点检测使用的设备
        landmarks_subset: 要评估的特征点索引（None = 全部68个点）
    
    返回:
        (平均NIOE, 每帧NIOE列表) 的元组
    """
    # 如果未提供检测器，则初始化
    if detector is None:
        try:
            import face_alignment
            detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                flip_input=False
            )
        except ImportError:
            raise ImportError(
                "未安装 face-alignment。使用以下命令安装: pip install face-alignment"
            )
    
    # 如果提供的是路径，则加载图像
    if isinstance(gen_images, list) and isinstance(gen_images[0], str):
        gen_images_paths = gen_images
        gt_images_paths = gt_images
    else:
        gen_images_paths = None
        gt_images_paths = None
    
    nioe_scores = []
    failed_frames = []
    
    for i, (gen, gt) in enumerate(zip(gen_images, gt_images)):
        # 如果需要则加载图像
        if gen_images_paths is not None:
            gen_img = cv2.imread(gen)
            gt_img = cv2.imread(gt)
            if gen_img is None or gt_img is None:
                print(f"警告: 加载索引 {i} 的图像失败")
                failed_frames.append(i)
                continue
            gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        else:
            gen_img = gen
            gt_img = gt
        
        try:
            # 检测特征点
            gen_landmarks = detector.get_landmarks(gen_img)
            gt_landmarks = detector.get_landmarks(gt_img)
            
            # 检查是否检测到人脸
            if gen_landmarks is None or gt_landmarks is None:
                print(f"警告: 索引 {i} 未检测到人脸")
                failed_frames.append(i)
                continue
            
            # 如果检测到多个人脸则使用第一个
            gen_lms = gen_landmarks[0]  # 形状: (68, 2)
            gt_lms = gt_landmarks[0]
            
            # 从真值计算眼间距
            # 左眼: 特征点 36-41, 右眼: 特征点 42-47
            left_eye_center = np.mean(gt_lms[36:42], axis=0)
            right_eye_center = np.mean(gt_lms[42:48], axis=0)
            iod = np.linalg.norm(left_eye_center - right_eye_center)
            
            # 如果指定了特征点子集，则选择
            if landmarks_subset is not None:
                gen_lms = gen_lms[landmarks_subset]
                gt_lms = gt_lms[landmarks_subset]
            
            # 计算归一化误差
            landmark_errors = np.linalg.norm(gen_lms - gt_lms, axis=1)
            mean_error = np.mean(landmark_errors)
            normalized_error = mean_error / iod
            
            nioe_scores.append(normalized_error)
            
        except Exception as e:
            print(f"警告: 处理帧 {i} 时出错: {e}")
            failed_frames.append(i)
            continue
    
    if len(nioe_scores) == 0:
        raise ValueError("没有成功处理的帧!")
    
    if len(failed_frames) > 0:
        print(f"警告: {len(failed_frames)} 帧处理失败")
    
    mean_nioe = np.mean(nioe_scores)
    
    return mean_nioe, nioe_scores


def compute_landmark_metrics(gen_dir: str,
                             gt_dir: str,
                             device: str = 'cuda',
                             subset: str = 'all') -> dict:
    """
    计算综合特征点指标
    
    参数:
        gen_dir: 包含生成图像的目录
        gt_dir: 包含真值图像的目录
        device: 特征点检测使用的设备
        subset: 要评估的特征点集合 ('all', 'mouth', 'eyes', 'contour')
    
    返回:
        包含特征点指标的字典
    """
    # 定义特征点子集 (68点模型索引)
    landmark_subsets = {
        'all': None,  # 全部 68 个点
        'mouth': list(range(48, 68)),  # 嘴部区域（内外唇）
        'eyes': list(range(36, 48)),   # 双眼
        'contour': list(range(0, 17)),  # 面部轮廓
        'nose': list(range(27, 36)),    # 鼻子
        'eyebrows': list(range(17, 27)) # 眉毛
    }
    
    if subset not in landmark_subsets:
        raise ValueError(f"无效的子集: {subset}。请从 {list(landmark_subsets.keys())} 中选择")
    
    # 获取图像路径
    gen_images = sorted([str(p) for p in Path(gen_dir).glob('*.jpg')])
    gt_images = sorted([str(p) for p in Path(gt_dir).glob('*.jpg')])
    
    min_len = min(len(gen_images), len(gt_images))
    gen_images = gen_images[:min_len]
    gt_images = gt_images[:min_len]
    
    print(f"为 {len(gen_images)} 对图像计算 NIOE...")
    print(f"特征点子集: {subset}")
    
    # 初始化检测器一次
    try:
        import face_alignment
        detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False
        )
    except ImportError:
        raise ImportError(
            "未安装 face-alignment。使用以下命令安装: pip install face-alignment"
        )
    
    # 计算 NIOE
    mean_nioe, nioe_list = compute_nioe(
        gen_images,
        gt_images,
        detector=detector,
        device=device,
        landmarks_subset=landmark_subsets[subset]
    )
    
    results = {
        'nioe': {
            'mean': mean_nioe,
            'std': np.std(nioe_list),
            'per_frame': nioe_list
        },
        'subset': subset,
        'num_frames': len(gen_images),
        'num_processed': len(nioe_list)
    }
    
    return results


def extract_and_save_landmarks(image_dir: str,
                               output_path: str,
                               device: str = 'cuda') -> int:
    """
    从所有图像中提取特征点并保存到文件
    用于预处理以避免在评估期间重新检测
    
    参数:
        image_dir: 包含图像的目录
        output_path: 保存特征点的路径（.npy 文件）
        device: 特征点检测使用的设备
    
    返回:
        处理的图像数量
    """
    try:
        import face_alignment
        detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False
        )
    except ImportError:
        raise ImportError(
            "未安装 face-alignment。使用以下命令安装: pip install face-alignment"
        )
    
    image_paths = sorted([str(p) for p in Path(image_dir).glob('*.jpg')])
    all_landmarks = []
    
    print(f"从 {len(image_paths)} 张图像中提取特征点...")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        landmarks = detector.get_landmarks(img)
        
        if landmarks is None:
            print(f"警告: 在 {img_path} 中未检测到人脸")
            all_landmarks.append(None)
        else:
            all_landmarks.append(landmarks[0])  # 使用第一个人脸
    
    # 保存特征点
    np.save(output_path, all_landmarks, allow_pickle=True)
    print(f"已保存特征点到 {output_path}")
    
    return len(image_paths)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试特征点指标')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='生成图像目录')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='真值图像目录')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--subset', type=str, default='all',
                        choices=['all', 'mouth', 'eyes', 'contour', 'nose', 'eyebrows'])
    
    args = parser.parse_args()
    
    results = compute_landmark_metrics(args.gen_dir, args.gt_dir, args.device, args.subset)
    
    print("\n" + "="*50)
    print("特征点指标结果")
    print("="*50)
    print(f"NIOE ({results['subset']}): {results['nioe']['mean']:.4f} ± {results['nioe']['std']:.4f}")
    print(f"已处理: {results['num_processed']}/{results['num_frames']} 帧")
    print("="*50)
