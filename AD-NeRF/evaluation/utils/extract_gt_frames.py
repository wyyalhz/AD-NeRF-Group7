"""
Extract ground truth frames from the original video.

This script extracts frames from the original video that correspond to the test set,
creating ground truth images for evaluation.
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_frame_index_from_data(frame_data):
    """Extract frame index from frame data (handles both file_path and img_id)."""
    if 'img_id' in frame_data:
        return frame_data['img_id']
    elif 'file_path' in frame_data:
        file_path = frame_data['file_path']
        frame_name = os.path.basename(file_path)
        return int(os.path.splitext(frame_name)[0])
    else:
        raise ValueError(f"Frame data has neither 'img_id' nor 'file_path': {frame_data.keys()}")


def extract_ground_truth_frames(video_path: str, 
                                 transforms_json: str, 
                                 output_dir: str,
                                 testskip: int = 1,
                                 max_frames: int = -1) -> int:
    """
    Extract ground truth frames from video based on transforms file.
    
    Args:
        video_path: Path to the original video file
        transforms_json: Path to transforms_val.json or transforms_train.json
        output_dir: Directory to save extracted frames
        testskip: Frame skip factor (1 = every frame, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to extract (-1 = all)
    
    Returns:
        Number of frames extracted
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transforms file
    with open(transforms_json, 'r') as f:
        transforms = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_video_frames} frames @ {fps} fps")
    print(f"Transforms file contains {len(transforms['frames'])} frame entries")
    
    # Extract frame indices from transforms
    frame_indices = []
    for i, frame_data in enumerate(transforms['frames']):
        if i % testskip == 0:
            # Get frame index (handles both file_path and img_id)
            frame_idx = get_frame_index_from_data(frame_data)
            frame_indices.append(frame_idx)
            
            if max_frames > 0 and len(frame_indices) >= max_frames:
                break
    
    print(f"Extracting {len(frame_indices)} frames...")
    
    # Extract frames
    extracted_count = 0
    for i, frame_idx in enumerate(tqdm(frame_indices)):
        # Set video position to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Save frame (use sequential numbering for output to match test_aud_rst)
        output_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_count += 1
    
    cap.release()
    print(f"Successfully extracted {extracted_count} frames to {output_dir}")
    
    return extracted_count


def extract_gt_from_ori_imgs(ori_imgs_dir: str,
                             transforms_json: str,
                             output_dir: str,
                             testskip: int = 1,
                             max_frames: int = -1) -> int:
    """
    Extract ground truth frames from pre-processed ori_imgs directory.
    This is useful if you've already extracted frames during preprocessing.
    
    Args:
        ori_imgs_dir: Directory containing preprocessed original images (dataset/{id}/ori_imgs)
        transforms_json: Path to transforms_val.json
        output_dir: Directory to save GT frames for evaluation
        testskip: Frame skip factor
        max_frames: Maximum number of frames to extract (-1 = all)
    
    Returns:
        Number of frames copied
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transforms file
    with open(transforms_json, 'r') as f:
        transforms = json.load(f)
    
    print(f"Transforms file contains {len(transforms['frames'])} frame entries")
    
    # Copy frames according to transforms
    copied_count = 0
    for i, frame_data in enumerate(transforms['frames']):
        if i % testskip == 0:
            # Get frame index (handles both file_path and img_id)
            frame_idx = get_frame_index_from_data(frame_data)
            
            # Source path in ori_imgs
            src_path = os.path.join(ori_imgs_dir, f"{frame_idx}.jpg")
            
            # Check if source exists
            if not os.path.exists(src_path):
                print(f"Warning: Source frame not found: {src_path}")
                continue
            
            # Destination path (sequential numbering)
            dst_path = os.path.join(output_dir, f"{copied_count}.jpg")
            
            # Copy frame
            img = cv2.imread(src_path)
            if img is None:
                print(f"Warning: Failed to read {src_path}")
                continue
            cv2.imwrite(dst_path, img)
            copied_count += 1
            
            if max_frames > 0 and copied_count >= max_frames:
                break
    
    print(f"Successfully copied {copied_count} frames to {output_dir}")
    return copied_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ground truth frames for evaluation')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject name (e.g., Obama, Jae-in)')
    parser.add_argument('--dataset_dir', type=str, default='../dataset',
                        help='Path to dataset directory')
    parser.add_argument('--use_video', action='store_true',
                        help='Extract from original video (default: use ori_imgs)')
    parser.add_argument('--testskip', type=int, default=1,
                        help='Frame skip factor')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Maximum frames to extract (-1 = all)')
    
    args = parser.parse_args()
    
    subject_dir = os.path.join(args.dataset_dir, args.subject)
    transforms_json = os.path.join(subject_dir, 'transforms_val.json')
    output_dir = os.path.join(subject_dir, 'gt_frames')
    
    if args.use_video:
        # Extract from video
        video_path = os.path.join(args.dataset_dir, 'vids', f'{args.subject}.mp4')
        extract_ground_truth_frames(
            video_path=video_path,
            transforms_json=transforms_json,
            output_dir=output_dir,
            testskip=args.testskip,
            max_frames=args.max_frames
        )
    else:
        # Copy from ori_imgs (faster and maintains consistency with preprocessing)
        ori_imgs_dir = os.path.join(subject_dir, 'ori_imgs')
        extract_gt_from_ori_imgs(
            ori_imgs_dir=ori_imgs_dir,
            transforms_json=transforms_json,
            output_dir=output_dir,
            testskip=args.testskip,
            max_frames=args.max_frames
        )
    
    print(f"\nGround truth frames ready for evaluation!")
    print(f"Generated images: {subject_dir}/logs/{args.subject}_com/test_aud_rst/")
    print(f"Ground truth images: {output_dir}/")
