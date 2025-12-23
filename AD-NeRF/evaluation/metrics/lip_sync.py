"""
Lip sync metrics: LSE-C and LSE-D (Lip Sync Error - Confidence & Distance).

These metrics evaluate audio-visual synchronization using SyncNet.
Requires the SyncNet model from Wav2Lip repository.
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Union
import subprocess


def compute_lse(video_path: str,
                audio_path: str,
                syncnet_checkpoint: str,
                device: str = 'cuda',
                use_wav2lip_eval: bool = True) -> Tuple[float, float]:
    """
    Compute Lip Sync Error metrics (LSE-C and LSE-D) using SyncNet.
    
    LSE-C (Confidence): Higher values indicate better sync (typically 0-10+)
    LSE-D (Distance): Lower values indicate better sync (typically 5-15)
    
    Args:
        video_path: Path to video file (.avi or .mp4)
        audio_path: Path to audio file (.wav)
        syncnet_checkpoint: Path to SyncNet checkpoint
        device: Device for computation
        use_wav2lip_eval: Whether to use Wav2Lip's evaluation script
    
    Returns:
        Tuple of (lse_confidence, lse_distance)
    """
    if use_wav2lip_eval:
        # Use Wav2Lip's evaluation script (recommended)
        return compute_lse_wav2lip(video_path, audio_path, syncnet_checkpoint, device)
    else:
        # Use custom implementation
        return compute_lse_custom(video_path, audio_path, syncnet_checkpoint, device)


def compute_lse_wav2lip(video_path: str,
                        audio_path: str,
                        syncnet_checkpoint: str,
                        device: str = 'cuda') -> Tuple[float, float]:
    """
    Compute LSE using SyncNet pipeline.

    Returns:
        Tuple of (lse_confidence, lse_distance)
    """
    import subprocess
    import glob
    import shutil
    import sys

    # Use syncnet_python's pipeline
    syncnet_path = Path(__file__).parent.parent / 'external' / 'syncnet_python'
    sys.path.insert(0, str(syncnet_path))

    # Create temp directory
    import hashlib
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    temp_data_dir = Path('/tmp') / f'syncnet_eval_{video_hash}'
    temp_data_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Run syncnet pipeline to preprocess video
        pipeline_script = syncnet_path / 'run_pipeline.py'

        cmd = [
            'python', str(pipeline_script),
            '--videofile', video_path,
            '--reference', 'eval',
            '--data_dir', str(temp_data_dir)
        ]

        print(f"Running SyncNet pipeline...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(syncnet_path))

        # Import SyncNetInstance
        from SyncNetInstance import SyncNetInstance

        # Setup options
        class Args:
            pass

        opt = Args()
        opt.initial_model = syncnet_checkpoint
        opt.batch_size = 20
        opt.vshift = 15
        opt.data_dir = str(temp_data_dir)
        opt.avi_dir = str(temp_data_dir / 'pyavi')
        opt.tmp_dir = str(temp_data_dir / 'pytmp')
        opt.work_dir = str(temp_data_dir / 'pywork')
        opt.crop_dir = str(temp_data_dir / 'pycrop')

        # Load model
        s = SyncNetInstance()
        s.loadParameters(opt.initial_model)

        # Get list of cropped videos
        flist = glob.glob(os.path.join(opt.crop_dir, 'eval', '0*.avi'))
        flist.sort()

        if not flist:
            print(f"Warning: No cropped videos found in {opt.crop_dir}/eval/")
            print(f"Pipeline output: {result.stdout}")
            print(f"Pipeline errors: {result.stderr}")
            raise ValueError(f"No cropped videos found in {opt.crop_dir}/eval/")

        # Evaluate each crop
        dists = []
        confs = []
        for fname in flist:
            offset, conf, dist = s.evaluate(opt, videofile=fname)
            dists.append(dist)
            confs.append(conf)

        # Return average scores (LSE-C, LSE-D)
        lse_c = float(np.mean(confs))
        lse_d = float(np.mean(dists))

        return lse_c, lse_d

    except subprocess.CalledProcessError as e:
        print(f"Error running SyncNet pipeline: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        # Cleanup
        if temp_data_dir.exists():
            shutil.rmtree(temp_data_dir, ignore_errors=True)



def compute_lse_custom(video_path: str,
                       audio_path: str,
                       syncnet_checkpoint: str,
                       device: str = 'cuda') -> Tuple[float, float]:
    """
    Custom implementation of LSE computation using SyncNet.
    
    This is a simplified implementation. For best results, use Wav2Lip's evaluation.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file  
        syncnet_checkpoint: Path to SyncNet checkpoint
        device: Device for computation
    
    Returns:
        Tuple of (lse_confidence, lse_distance)
    """
    # Load SyncNet model
    syncnet = load_syncnet_model(syncnet_checkpoint, device)
    
    # Extract video frames
    video_frames = extract_video_frames(video_path)
    
    # Extract audio features
    audio_features = extract_audio_features(audio_path)
    
    # Ensure same number of frames
    min_len = min(len(video_frames), len(audio_features))
    video_frames = video_frames[:min_len]
    audio_features = audio_features[:min_len]
    
    # Compute sync scores
    confidences = []
    distances = []
    
    batch_size = 32
    for i in range(0, len(video_frames), batch_size):
        batch_video = video_frames[i:i+batch_size]
        batch_audio = audio_features[i:i+batch_size]
        
        # Get embeddings
        video_emb = syncnet.forward_video(batch_video)
        audio_emb = syncnet.forward_audio(batch_audio)
        
        # Compute sync metrics
        for v_emb, a_emb in zip(video_emb, audio_emb):
            # Confidence: cosine similarity
            confidence = torch.nn.functional.cosine_similarity(
                v_emb.unsqueeze(0), a_emb.unsqueeze(0)
            ).item()
            confidences.append(confidence)
            
            # Distance: L2 distance
            distance = torch.norm(v_emb - a_emb, p=2).item()
            distances.append(distance)
    
    # Average scores
    lse_c = np.mean(confidences)
    lse_d = np.mean(distances)
    
    return lse_c, lse_d


def load_syncnet_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load SyncNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to SyncNet checkpoint
        device: Device to load model on
    
    Returns:
        Loaded SyncNet model
    """
    # This is a placeholder - actual implementation depends on SyncNet architecture
    # Users should use Wav2Lip's evaluation script instead
    raise NotImplementedError(
        "Custom SyncNet loading not implemented. "
        "Please use Wav2Lip's evaluation script by setting use_wav2lip_eval=True"
    )


def extract_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract frames from video for SyncNet processing.
    
    Args:
        video_path: Path to video file
    
    Returns:
        List of video frames (lip region crops)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # TODO: Crop to lip region
        frames.append(frame)
    
    cap.release()
    return frames


def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract audio features for SyncNet.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Audio features array
    """
    # This is a placeholder - actual implementation would use MFCC or mel spectrogram
    raise NotImplementedError(
        "Audio feature extraction not implemented. "
        "Please use Wav2Lip's evaluation script."
    )


def create_video_from_images(image_dir: str,
                             audio_path: str,
                             output_video: str,
                             fps: int = 25) -> str:
    """
    Create video from image directory and audio file for LSE evaluation.
    
    Args:
        image_dir: Directory containing sequential images
        audio_path: Path to audio file (.wav)
        output_video: Output video path
        fps: Frames per second
    
    Returns:
        Path to created video
    """
    import imageio
    
    # Get all images
    images = sorted(Path(image_dir).glob('*.jpg'))
    
    if len(images) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(images)} images...")
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        out.write(img)
    
    out.release()
    
    # Add audio using ffmpeg
    if os.path.exists(audio_path):
        output_with_audio = output_video.replace('.avi', '_audio.avi')
        cmd = [
            'ffmpeg', '-i', output_video, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
            output_with_audio, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Created video with audio: {output_with_audio}")
            return output_with_audio
        except subprocess.CalledProcessError:
            print("Warning: Could not add audio to video. Returning video without audio.")
            return output_video
    
    return output_video


def compute_lip_sync_metrics(gen_dir: str,
                             audio_path: str,
                             syncnet_checkpoint: str,
                             device: str = 'cuda',
                             fps: int = 25) -> dict:
    """
    Compute lip sync metrics for generated video.
    
    Args:
        gen_dir: Directory containing generated images
        audio_path: Path to audio file
        syncnet_checkpoint: Path to SyncNet checkpoint
        device: Device for computation
        fps: Video frame rate
    
    Returns:
        Dictionary with lip sync metrics
    """
    # Create temporary video from images
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_video = os.path.join(temp_dir, 'temp_video.avi')
    
    print("Creating video from generated images...")
    video_path = create_video_from_images(gen_dir, audio_path, temp_video, fps)
    
    print("Computing lip sync metrics...")
    try:
        lse_c, lse_d = compute_lse(
            video_path, audio_path, syncnet_checkpoint, device
        )
        
        results = {
            'lse_confidence': lse_c,
            'lse_distance': lse_d,
            'video_path': video_path
        }
        
        return results
        
    except Exception as e:
        print(f"Error computing lip sync metrics: {e}")
        # Clean up
        if os.path.exists(temp_video):
            os.remove(temp_video)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test lip sync metrics')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='Directory with generated images')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--syncnet_checkpoint', type=str, required=True,
                        help='Path to SyncNet checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--fps', type=int, default=25,
                        help='Video frame rate')
    
    args = parser.parse_args()
    
    results = compute_lip_sync_metrics(
        args.gen_dir,
        args.audio,
        args.syncnet_checkpoint,
        args.device,
        args.fps
    )
    
    print("\n" + "="*50)
    print("Lip Sync Metrics Results")
    print("="*50)
    print(f"LSE-C (Confidence): {results['lse_confidence']:.4f}")
    print(f"LSE-D (Distance):   {results['lse_distance']:.4f}")
    print(f"Video saved to: {results['video_path']}")
    print("="*50)
