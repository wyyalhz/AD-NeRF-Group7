"""
Data loader utilities for evaluation.

This module provides functions to load and pair generated images with ground truth images.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import cv2


def load_image_pairs(generated_dir: str, gt_dir: str, extension: str = 'jpg') -> List[Tuple[str, str]]:
    """
    Load pairs of generated and ground truth images.
    
    Args:
        generated_dir: Directory containing generated images
        gt_dir: Directory containing ground truth images
        extension: Image file extension (default: 'jpg')
    
    Returns:
        List of tuples (generated_path, gt_path)
    """
    generated_dir = Path(generated_dir)
    gt_dir = Path(gt_dir)
    
    # Get all generated images
    gen_images = sorted(generated_dir.glob(f'*.{extension}'))
    
    pairs = []
    for gen_img in gen_images:
        # Find corresponding GT image with same name
        gt_img = gt_dir / gen_img.name
        if gt_img.exists():
            pairs.append((str(gen_img), str(gt_img)))
        else:
            print(f"Warning: Ground truth not found for {gen_img.name}")
    
    return pairs


def get_test_frames(config_path: str, dataset_dir: str) -> Dict:
    """
    Get test frame information from config and transforms file.
    
    Args:
        config_path: Path to TorsoNeRFTest_config.txt
        dataset_dir: Path to dataset directory (e.g., dataset/Obama)
    
    Returns:
        Dictionary with test frame information
    """
    dataset_dir = Path(dataset_dir)
    
    # Read config to get test pose file
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    
    # Load transforms file
    test_pose_file = config.get('test_pose_file', 'transforms_val.json')
    transforms_path = dataset_dir / test_pose_file
    
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Extract frame information
    frames_info = {
        'focal_len': float(transforms['focal_len']),
        'cx': float(transforms['cx']),
        'cy': float(transforms['cy']),
        'frames': []
    }
    
    for frame in transforms['frames']:
        frame_info = {
            'file_path': frame['file_path'],
            'transform_matrix': frame['transform_matrix']
        }
        frames_info['frames'].append(frame_info)
    
    return frames_info


def load_images_from_paths(image_paths: List[str], normalize: bool = True) -> np.ndarray:
    """
    Load multiple images from file paths.
    
    Args:
        image_paths: List of image file paths
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Numpy array of shape (N, H, W, 3)
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load image {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if normalize:
            img = img.astype(np.float32) / 255.0
        images.append(img)
    
    return np.array(images)


def get_frame_index_from_filename(filename: str) -> int:
    """
    Extract frame index from filename (e.g., '042.jpg' -> 42).
    
    Args:
        filename: Image filename
    
    Returns:
        Frame index as integer
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(basename)
    except ValueError:
        print(f"Warning: Could not extract frame index from {filename}")
        return -1


def validate_image_pairs(pairs: List[Tuple[str, str]], check_size: bool = True) -> List[Tuple[str, str]]:
    """
    Validate that image pairs exist and have matching dimensions.
    
    Args:
        pairs: List of (generated_path, gt_path) tuples
        check_size: Whether to check if images have matching dimensions
    
    Returns:
        List of valid pairs
    """
    valid_pairs = []
    
    for gen_path, gt_path in pairs:
        # Check if files exist
        if not os.path.exists(gen_path):
            print(f"Warning: Generated image not found: {gen_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth image not found: {gt_path}")
            continue
        
        # Check dimensions if requested
        if check_size:
            gen_img = cv2.imread(gen_path)
            gt_img = cv2.imread(gt_path)
            
            if gen_img is None or gt_img is None:
                print(f"Warning: Failed to load images for pair")
                continue
            
            if gen_img.shape != gt_img.shape:
                print(f"Warning: Image size mismatch for {os.path.basename(gen_path)}: "
                      f"{gen_img.shape} vs {gt_img.shape}")
                continue
        
        valid_pairs.append((gen_path, gt_path))
    
    return valid_pairs


if __name__ == '__main__':
    # Test the data loader
    test_dir = "../AD-NeRF/dataset/Obama/logs/Obama_com/test_aud_rst"
    gt_dir = "../AD-NeRF/dataset/Obama/gt_frames"
    
    if os.path.exists(test_dir):
        pairs = load_image_pairs(test_dir, gt_dir)
        print(f"Found {len(pairs)} image pairs")
        
        if pairs:
            print(f"Example pair: {pairs[0]}")
