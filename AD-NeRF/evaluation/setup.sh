#!/bin/bash
# Setup script for AD-NeRF Evaluation Module
# Run this after creating your conda environment

echo "=========================================="
echo "AD-NeRF Evaluation Module Setup"
echo "=========================================="

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r evaluation/requirements.txt

# Check if Wav2Lip should be cloned
echo ""
echo "🎤 Setting up Wav2Lip for lip sync metrics..."
read -p "Do you want to clone Wav2Lip for LSE metrics? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    if [ -d "evaluation/external/Wav2Lip" ]; then
        echo "✓ Wav2Lip already exists"
    else
        cd evaluation/external
        git clone https://github.com/Rudrabha/Wav2Lip.git
        cd Wav2Lip
        pip install -r requirements.txt
        cd ../../..
        echo "✓ Wav2Lip installed"
    fi
else
    echo "⚠ Skipping Wav2Lip (LSE metrics will not be available)"
fi

# Create sample evaluation script
echo ""
echo "📝 Creating sample evaluation script..."
cat > evaluation/run_example.sh << 'EOF'
#!/bin/bash
# Example evaluation script for Obama dataset

echo "Running evaluation on Obama dataset..."

python evaluation/evaluate.py \
    --subject Obama \
    --metrics psnr ssim fid \
    --device cuda \
    --save_json

echo "Evaluation complete! Check evaluation/results/Obama/ for results."
EOF

chmod +x evaluation/run_example.sh

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Extract ground truth frames:"
echo "   python evaluation/utils/extract_gt_frames.py --subject Obama"
echo ""
echo "2. Run evaluation:"
echo "   python evaluation/evaluate.py --subject Obama --metrics psnr ssim fid"
echo ""
echo "3. Or use the example script:"
echo "   bash evaluation/run_example.sh"
echo ""
echo "For more information, see evaluation/README.md"
echo "=========================================="
