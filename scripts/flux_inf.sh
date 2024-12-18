#!/bin/bash

# 设置常用的参数
CKPT_PATH='models/flux/flux1-dev.sft'
CLIP_L='models/flux/clip_l.safetensors'
T5XXL='models/flux/t5xxl_fp16.safetensors'
AE='models/flux/ae.sft'
DTYPE='bf16'
FLUX_DTYPE="fp8"
LORA_PATH='models/lora/fluxlora/chinese_painting-30.safetensors;1.0'
WIDTH=1024
HEIGHT=1024
STEPS=50
GUIDANCE=3.5
CFG_SCALE=1.0
N_SAMPLES=50
OUTPUT_DIR_PREFIX='benc/flux/fluxlora_40'

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
    
    python modules/dev/flux_minimal_inference.py \
        --lora_weights "$LORA_PATH" \
        --ckpt "$CKPT_PATH" \
        --clip_l "$CLIP_L" \
        --ae "$AE" \
        --dtype "$DTYPE" \
        --flux_dtype "$FLUX_DTYPE" \
        --guidance $GUIDANCE \
        --cfg_scale $CFG_SCALE \
        --t5xxl "$T5XXL" \
        --prompt "${PROMPTS[$i]}" \
        --negative_prompt '' \
        --output "$OUTPUT_DIR" \
        --offload \
        --width $WIDTH \
        --height $HEIGHT \
        --steps $STEPS \
        --n_samples $N_SAMPLES
done

echo "All tasks completed."
