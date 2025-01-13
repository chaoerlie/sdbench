#!/bin/bash

# 设置常用的参数
CKPT_PATH='models/sd3/sd3.5_medium.safetensors'
CLIP_G='models/sd3/clip_g.safetensors'
CLIP_L='models/flux/clip_l.safetensors'
T5XXL='models/flux/t5xxl_fp16.safetensors'
LORA_PATH='train/sd3/models/chinese_painting.safetensors;1.0'
WIDTH=1024
HEIGHT=1024
STEPS=50
N_SAMPLES=50
OUTPUT_DIR_PREFIX='benc/sd3/SD3lora_50'

# 定义每个任务的提示词和输出目录
declare -a PROMPTS=(
    "chinese painting, Shanshui, Majestic mountain peaks draped in mist, towering pine trees, and cascading waterfalls, creating a serene atmosphere with soft ink wash and traditional brushwork."
    "chinese painting, Shanshui, A winding river flowing peacefully through ancient mountains, surrounded by mist, with delicate brush strokes creating a harmonious and tranquil landscape."
    "chinese painting, Shanshui, Misty cliffs rising over a winding river, where the fog gently envelops the scene, soft gradients of ink creating a sense of peaceful isolation."
    "chinese painting, Shanshui, Distant mountains shrouded in mist, a small boat drifting along a fog-covered river, delicate brushwork capturing the quietude and stillness of the scene."
    "chinese painting, Shanshui, An ancient stone bridge crossing a serene stream, surrounded by dense forests, with towering mountains in the backdrop, painted in traditional Chinese style."
    "chinese painting, Shanshui, A lone fisherman on a small boat, drifting along a mist-covered river, mountains rising on either side, soft fading ink tones evoking a tranquil and peaceful mood."
    "chinese painting, Shanshui, A vibrant sunset over a tranquil lake, with majestic mountains in the background, ink wash and intricate brushwork capturing the warmth and serenity of the moment."
    "chinese painting, Shanshui, A winding path through a bamboo forest, leading to a distant mountain, with mist rising between the bamboo stalks and soft brushstrokes evoking tranquility."
    "chinese painting, Shanshui, An ancient pagoda perched on a mountain peak, surrounded by mist and pine trees, rendered in soft ink wash with detailed brushwork capturing the classical style."
    "chinese painting, Shanshui, A peaceful landscape featuring a river winding through high mountain peaks, mist swirling over the water, soft ink wash and gentle contrasts creating a serene atmosphere."
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
