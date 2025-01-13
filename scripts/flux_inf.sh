#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 设置常用的参数
CKPT_PATH='models/flux/flux1-dev.sft'
CLIP_L='models/flux/clip_l.safetensors'
T5XXL='models/flux/t5xxl_fp16.safetensors'
AE='models/flux/ae.sft'
DTYPE='bf16'
FLUX_DTYPE="fp8"
LORA_PATH='/home/ps/sdbench/models/lora/fluxlora/chinese_painting_new.safetensors;1.0'
WIDTH=1024
HEIGHT=1024
STEPS=50
GUIDANCE=3.5
CFG_SCALE=1.0
N_SAMPLES=50
OUTPUT_DIR_PREFIX='benc/flux/fluxlora_50'

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
