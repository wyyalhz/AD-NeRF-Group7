#!/bin/bash
# Batch evaluation script for multiple subjects
# Usage: bash evaluation/batch_evaluate.sh [metrics]
# Example: bash evaluation/batch_evaluate.sh psnr ssim fid

# Default metrics if not specified
METRICS=${@:-psnr ssim fid}

echo "=========================================="
echo "Batch Evaluation for All Subjects"
echo "Metrics: $METRICS"
echo "=========================================="

# List of subjects (modify as needed)
SUBJECTS=("Obama" "Jae-in" "Lieu" "Macron")

# Results summary file
SUMMARY_FILE="AD-NeRF/evaluation/results/batch_summary.txt"
mkdir -p AD-NeRF/evaluation/results

echo "Batch Evaluation Summary" > $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "Metrics: $METRICS" >> $SUMMARY_FILE
echo "========================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Evaluate each subject
for subject in "${SUBJECTS[@]}"; do
    echo ""
    echo "🎬 Evaluating: $subject"
    echo "=========================================="
    
    # Check if subject exists
    if [ ! -d "AD-NeRF/dataset/$subject" ]; then
        echo "⚠ Warning: AD-NeRF/dataset/$subject not found. Skipping..."
        echo "$subject: NOT FOUND" >> $SUMMARY_FILE
        continue
    fi
    
    # Run evaluation
    python AD-NeRF/evaluation/evaluate.py \
        --subject $subject \
        --metrics $METRICS \
        --device cuda \
        --save_json \
        --skip_gt_extraction
    
    if [ $? -eq 0 ]; then
        echo "✓ $subject completed successfully"
        
        # Extract metrics from report
        REPORT_FILE="AD-NeRF/evaluation/results/$subject/evaluation_report.txt"
        if [ -f "$REPORT_FILE" ]; then
            echo "" >> $SUMMARY_FILE
            echo "Subject: $subject" >> $SUMMARY_FILE
            grep "PSNR:" $REPORT_FILE >> $SUMMARY_FILE 2>/dev/null
            grep "SSIM:" $REPORT_FILE >> $SUMMARY_FILE 2>/dev/null
            grep "FID:" $REPORT_FILE >> $SUMMARY_FILE 2>/dev/null
            grep "NIOE:" $REPORT_FILE >> $SUMMARY_FILE 2>/dev/null
            grep "LSE-C" $REPORT_FILE >> $SUMMARY_FILE 2>/dev/null
            grep "LSE-D" $REPORT_FILE >> $SUMMARY_FILE 2>/dev/null
        fi
    else
        echo "✗ $subject failed"
        echo "$subject: FAILED" >> $SUMMARY_FILE
    fi
done

echo ""
echo "=========================================="
echo "✓ Batch evaluation complete!"
echo "Summary saved to: $SUMMARY_FILE"
echo "=========================================="
echo ""
cat $SUMMARY_FILE
