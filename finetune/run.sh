#!/bin/bash
# Fine-tune OPUS-MT models for bidirectional translation (zh↔en)
# Run on Linux server with GPU

set -e

# Install dependencies
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Set HF mirror for China
export HF_ENDPOINT=https://hf-mirror.com

echo "============================================"
echo "Fine-tuning both directions (zh→en + en→zh)"
echo "============================================"

# Both directions (uses 2M samples each by default)
python train.py --direction both \
    --max_samples 2000000 \
    --batch_size 8 \
    --grad_accum 4 \
    --epochs 3 \
    --fp16 \
    --use_lora

echo ""
echo "============================================"
echo "Done! Models saved to:"
echo "  zh→en: finetune/output_zh2en/"
echo "  en→zh: finetune/output_en2zh/"
echo "============================================"
