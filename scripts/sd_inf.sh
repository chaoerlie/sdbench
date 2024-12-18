#!/bin/bash

# 设置常用的参数
CKPT_PATH="/home/ps/sdbench/models/1.ckpt"
OUTPUT_PREFIX="benc/sdxl/sdxllora_50"

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
    python modules/stable/gen_img.py \
        --v1 \
        --prompt "${PROMPTS[$i]}" \
        --ckpt "$CKPT_PATH" \
        --steps 50 \
        --images_per_prompt  50 \
        --sampler "k_dpm_2" \
        --scale 7.5 \
        --outdir "$OUTPUT_DIR" \
        --W 512 \
        --H 512 \
        --batch_size 1 \
        --clip_skip 2 \
        --fp16  \
        --xformers \
        --network_module "networks.lora" \
        --network_weights "/home/ps/sdbench/models/lora/sdlora/chinese_painting-50.safetensors" \
        --network_mul 1
done

echo "All tasks completed."
