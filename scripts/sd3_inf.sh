#!/bin/bash

# 设置常用的参数
CKPT_PATH='models/sd3/sd3.5_medium.safetensors'
CLIP_G='models/sd3/clip_g.safetensors'
CLIP_L='models/flux/clip_l.safetensors'
T5XXL='models/flux/t5xxl_fp16.safetensors'
LORA_PATH='models/lora/sd3lora/chinese_painting-40.safetensors;1.0'
WIDTH=1024
HEIGHT=1024
STEPS=50
N_SAMPLES=50
OUTPUT_DIR_PREFIX='benc/sd3/SD3lora_40'

# 定义每个任务的提示词和输出目录
declare -a PROMPTS=(
    "chinese_painting, lotus flowers, with soft brushstrokes showing the flowers floating on a calm pond, surrounded by green leaves and some mist."
    "chinese_painting, mountains and rivers, with towering peaks, flowing water, and scattered trees"
    "chinese_painting, a tiger, bold brushstrokes, standing in a natural setting surrounded by rocks and tall grass."
    "chinese_painting, flowers and birds, with vibrant blossoms and delicate birds perched among the branches."
    "chinese_painting, mountains and a waterfall, with towering mountains, flowing water, and a waterfall cascading down into a stream, surrounded by trees and rocks."
)

# 循环执行每个命令
for i in "${!PROMPTS[@]}"; do
    OUTPUT_DIR="${OUTPUT_DIR_PREFIX}_$((i + 1))"
    
    python modules/dev/sd3_minimal_inference.py \
        --lora_weights "$LORA_PATH" \
        --ckpt "$CKPT_PATH" \
        --clip_g "$CLIP_G" \
        --clip_l "$CLIP_L" \
        --t5xxl "$T5XXL" \
        --prompt "${PROMPTS[$i]}" \
        --negative_prompt '' \
        --output "$OUTPUT_DIR" \
        --offload \
        --bf16 \
        --width $WIDTH \
        --height $HEIGHT \
        --steps $STEPS \
        --n_samples $N_SAMPLES
done

echo "All tasks completed."
