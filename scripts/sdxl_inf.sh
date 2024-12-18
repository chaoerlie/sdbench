#!/bin/bash

# 设置常用的参数
CKPT_PATH="/home/ps/sdbench/models/sd_xl.safetensors"
LORA_PATH="/home/ps/sdbench/models/lora/sdxl/chinese_painting-40.safetensors;1.0"
GUIDANCE_SCALE=7.5
STEPS=50
N_SAMPLES=50
OUTPUT_PREFIX="benc/sdxl/sdxllora_50"
HEIGHT=1024
WIDTH=1024

# 定义每个任务的提示词
declare -a PROMPTS=(
    "chinese_painting, lotus flowers, with soft brushstrokes showing the flowers floating on a calm pond, surrounded by green leaves and some mist."
    "chinese_painting, mountains and rivers, with towering peaks, flowing water, and scattered trees."
    "chinese_painting, a tiger, bold brushstrokes, standing in a natural setting surrounded by rocks and tall grass."
    "chinese_painting, flowers and birds, with vibrant blossoms and delicate birds perched among the branches."
    "chinese_painting, mountains and a waterfall, with towering mountains, flowing water, and a waterfall cascading down into a stream, surrounded by trees and rocks."
)

# 循环执行每个命令
for i in "${!PROMPTS[@]}"; do
    # 动态构建输出目录
    OUTPUT_DIR="${OUTPUT_PREFIX}_$((i + 1))"
    
    # 执行推理命令
    python modules/stable/sdxl_minimal_inference.py \
        --prompt "${PROMPTS[$i]}" \
        --ckpt "$CKPT_PATH" \
        --steps $STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --output_dir "$OUTPUT_DIR" \
        --n_samples $N_SAMPLES \
        --lora_weights "$LORA_PATH" \
        --height $HEIGHT \
        --width $WIDTH
done

echo "All tasks completed."
