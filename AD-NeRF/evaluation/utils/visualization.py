"""
Visualization utilities for evaluation metrics.

Create plots and comparison figures for metric results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import json


def plot_metrics(results: Dict,
                 output_path: str,
                 title: Optional[str] = None,
                 figsize: tuple = (12, 8)) -> None:
    """
    Create comprehensive visualization of all metrics.
    
    Args:
        results: Dictionary containing metric results
        output_path: Path to save the plot
        title: Plot title (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title or 'Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: PSNR over frames
    if 'psnr' in results and 'per_frame' in results['psnr']:
        ax = axes[0, 0]
        psnr_scores = results['psnr']['per_frame']
        frames = range(len(psnr_scores))
        
        ax.plot(frames, psnr_scores, 'b-', linewidth=1, alpha=0.6, label='PSNR')
        ax.axhline(y=results['psnr']['mean'], color='r', linestyle='--', 
                   label=f"Mean: {results['psnr']['mean']:.2f} dB")
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Peak Signal-to-Noise Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: SSIM over frames
    if 'ssim' in results and 'per_frame' in results['ssim']:
        ax = axes[0, 1]
        ssim_scores = results['ssim']['per_frame']
        frames = range(len(ssim_scores))
        
        ax.plot(frames, ssim_scores, 'g-', linewidth=1, alpha=0.6, label='SSIM')
        ax.axhline(y=results['ssim']['mean'], color='r', linestyle='--',
                   label=f"Mean: {results['ssim']['mean']:.4f}")
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Plot 3: NIOE over frames (if available)
    if 'nioe' in results and 'per_frame' in results['nioe']:
        ax = axes[1, 0]
        nioe_scores = results['nioe']['per_frame']
        frames = range(len(nioe_scores))
        
        ax.plot(frames, nioe_scores, 'm-', linewidth=1, alpha=0.6, label='NIOE')
        ax.axhline(y=results['nioe']['mean'], color='r', linestyle='--',
                   label=f"Mean: {results['nioe']['mean']:.4f}")
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('NIOE')
        ax.set_title('Normalized Inter-Ocular Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'NIOE data not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Normalized Inter-Ocular Error')
    
    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    metrics_names = []
    metrics_values = []
    
    if 'psnr' in results:
        metrics_names.append('PSNR\n(dB)')
        metrics_values.append(results['psnr']['mean'])
    
    if 'ssim' in results:
        metrics_names.append('SSIM\n(×10)')
        metrics_values.append(results['ssim']['mean'] * 10)  # Scale for visibility
    
    if 'fid' in results and results['fid'] is not None:
        metrics_names.append('FID')
        metrics_values.append(results['fid'])
    
    if 'nioe' in results:
        metrics_names.append('NIOE\n(×100)')
        metrics_values.append(results['nioe']['mean'] * 100)  # Scale for visibility
    
    if metrics_names:
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(metrics_names)]
        ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Metrics Summary')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def save_metric_comparison(results_dict: Dict[str, Dict],
                           output_path: str,
                           metric: str = 'psnr',
                           figsize: tuple = (10, 6)) -> None:
    """
    Compare a specific metric across multiple subjects/experiments.
    
    Args:
        results_dict: Dictionary mapping subject names to their results
        output_path: Path to save the comparison plot
        metric: Which metric to compare ('psnr', 'ssim', 'fid', 'nioe')
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    subjects = list(results_dict.keys())
    values = []
    errors = []
    
    for subject in subjects:
        if metric in results_dict[subject]:
            metric_data = results_dict[subject][metric]
            if isinstance(metric_data, dict):
                values.append(metric_data.get('mean', 0))
                errors.append(metric_data.get('std', 0))
            else:
                values.append(metric_data)
                errors.append(0)
        else:
            values.append(0)
            errors.append(0)
    
    # Create bar plot
    x_pos = np.arange(len(subjects))
    ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    
    # Labels
    metric_labels = {
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'fid': 'FID (lower is better)',
        'nioe': 'NIOE (lower is better)'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric.upper()))
    ax.set_title(f'{metric.upper()} Comparison Across Subjects')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_frame_by_frame_comparison(gen_images: List[str],
                                   gt_images: List[str],
                                   psnr_scores: List[float],
                                   output_dir: str,
                                   num_samples: int = 5) -> None:
    """
    Create side-by-side comparison of generated vs GT images with PSNR scores.
    
    Args:
        gen_images: List of generated image paths
        gt_images: List of ground truth image paths
        psnr_scores: List of PSNR scores for each frame
        output_dir: Directory to save comparison images
        num_samples: Number of sample frames to visualize
    """
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select frames: best, worst, and median
    sorted_indices = np.argsort(psnr_scores)
    
    # Select diverse samples
    sample_indices = []
    if len(sorted_indices) > 0:
        # Worst
        sample_indices.append(sorted_indices[0])
        # Best
        sample_indices.append(sorted_indices[-1])
        # Median
        sample_indices.append(sorted_indices[len(sorted_indices) // 2])
        # Add random samples
        while len(sample_indices) < min(num_samples, len(sorted_indices)):
            idx = np.random.choice(sorted_indices)
            if idx not in sample_indices:
                sample_indices.append(idx)
    
    for idx in sample_indices:
        gen_img = cv2.imread(gen_images[idx])
        gt_img = cv2.imread(gt_images[idx])
        
        # Create comparison
        comparison = np.hstack([gen_img, gt_img])
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Generated', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Ground Truth', (gen_img.shape[1] + 10, 30), 
                   font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f'PSNR: {psnr_scores[idx]:.2f} dB', 
                   (10, comparison.shape[0] - 10), font, 1, (255, 255, 0), 2)
        
        # Save
        output_path = os.path.join(output_dir, f'comparison_frame_{idx:04d}.jpg')
        cv2.imwrite(output_path, comparison)
    
    print(f"Saved {len(sample_indices)} comparison images to {output_dir}")


def create_report_table(results: Dict, output_path: str) -> None:
    """
    Create a text report summarizing all metrics.
    
    Args:
        results: Dictionary containing all metric results
        output_path: Path to save the report
    """
    lines = []
    lines.append("="*60)
    lines.append("AD-NeRF Evaluation Report")
    lines.append("="*60)
    lines.append("")
    
    # Image Quality Metrics
    lines.append("Image Quality Metrics:")
    lines.append("-" * 40)
    
    if 'psnr' in results:
        lines.append(f"  PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
    
    if 'ssim' in results:
        lines.append(f"  SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
    
    if 'fid' in results and results['fid'] is not None:
        lines.append(f"  FID:  {results['fid']:.2f}")
    
    lines.append("")
    
    # Landmark Metrics
    if 'nioe' in results:
        lines.append("Landmark Metrics:")
        lines.append("-" * 40)
        lines.append(f"  NIOE: {results['nioe']['mean']:.4f} ± {results['nioe']['std']:.4f}")
        lines.append("")
    
    # Lip Sync Metrics
    if 'lse_confidence' in results or 'lse_distance' in results:
        lines.append("Lip Sync Metrics:")
        lines.append("-" * 40)
        if 'lse_confidence' in results:
            lines.append(f"  LSE-C (Confidence): {results['lse_confidence']:.4f}")
        if 'lse_distance' in results:
            lines.append(f"  LSE-D (Distance):   {results['lse_distance']:.4f}")
        lines.append("")
    
    # Summary
    lines.append("Summary:")
    lines.append("-" * 40)
    if 'num_frames' in results:
        lines.append(f"  Total frames evaluated: {results['num_frames']}")
    if 'subject' in results:
        lines.append(f"  Subject: {results['subject']}")
    
    lines.append("="*60)
    
    # Write to file
    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Saved report to {output_path}")
    
    # Also print to console
    print("\n" + report_text)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization utilities')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to evaluation results JSON file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for plots and reports')
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject name (for title)')
    
    args = parser.parse_args()
    
    # Load results from JSON
    with open(args.results_json, 'r') as f:
        results = json.load(f)
    
    # Add subject if provided
    if args.subject:
        results['subject'] = args.subject
    
    # Generate outputs
    output_plot = os.path.join(args.output_dir, 'metrics_plot.png')
    output_report = os.path.join(args.output_dir, 'evaluation_report.txt')
    
    title = f"{results.get('subject', 'AD-NeRF')} Evaluation"
    
    print("Generating visualizations...")
    plot_metrics(results, output_plot, title)
    create_report_table(results, output_report)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Plot: {output_plot}")
    print(f"  Report: {output_report}")
