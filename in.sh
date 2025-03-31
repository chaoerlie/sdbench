#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 设置常用的参数
CKPT_PATH='models/flux/flux1-dev.sft'
# CKPT_PATH='/data/flux_finetune/abstract_painting-000002.safetensors'
CLIP_L='models/flux/clip_l.safetensors'
T5XXL='models/flux/t5xxl_fp16.safetensors'
AE='models/flux/ae.sft'
AE='models/flux/ae.sft'
DTYPE='bf16'
FLUX_DTYPE="fp8"
LORA_PATH='/home/ps/sdbench/train/output/flux/Renwu/Renwu.safetensors;0.8'
WIDTH=1024
HEIGHT=1024
STEPS=50
GUIDANCE=3.5
CFG_SCALE=1.0
N_SAMPLES=20
OUTPUT_DIR_PREFIX='generation/flux/flux_Renwu_1'

declare -a PROMPTS=(
    "chinese painting, Renwu, A wise old scholar sitting under an ancient pine tree, reading a bamboo scroll with a cup of tea beside him."
    "chinese painting, Renwu, A wandering Taoist monk walking along a mountain path, carrying a gourd and a staff, with misty peaks in the background."
    "chinese painting, Renwu, A noble lady in flowing silk robes playing a guqin by a tranquil lake, cherry blossoms drifting in the wind."
    "chinese painting, Renwu, A fierce general in traditional armor standing on a battlefield, his long spear pointed forward, banners flying behind him."
    "chinese painting, Renwu, A calligrapher seated in a bamboo grove, dipping his brush in ink as he composes poetry on rice paper."
    "chinese painting, Renwu, A fisherman wearing a straw hat, casting his net into a misty river at sunrise, his boat barely visible in the fog."
    "chinese painting, Renwu, A group of poets gathered around a stone table in a garden, sharing wine and reciting verses under a full moon."
    "chinese painting, Renwu, A martial artist practicing a graceful sword dance atop a rocky cliff, with swirling clouds below."
    "chinese painting, Renwu, A mother and child sitting on the veranda of a traditional courtyard house, watching the autumn leaves fall."
    "chinese painting, Renwu, A legendary immortal riding a crane through the sky, with flowing robes and a serene expression."
)



# 循环执行每个命令
for i in "${!PROMPTS[@]}"; do
# for ((i=8; i<${#PROMPTS[@]}; i++)); do
    OUTPUT_DIR="${OUTPUT_DIR_PREFIX}_$((i + 1))"
    
    python modules/dev/flux_minimal_inference.py \
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
        --n_samples $N_SAMPLES \
        --lora_weights "$LORA_PATH" 
done

echo "All tasks completed 1."




CKPT_PATH='models/sd3/sd3.5_medium.safetensors'
CLIP_G='models/sd3/clip_g.safetensors'
CLIP_L='models/flux/clip_l.safetensors'
T5XXL='models/flux/t5xxl_fp16.safetensors'
LORA_PATH='/home/ps/sdbench/train/output/sd3/Renwu/Renwu.safetensors;0.8'
WIDTH=1024
HEIGHT=1024
STEPS=50
N_SAMPLES=20
OUTPUT_DIR_PREFIX='generation/sd3/SD3lora_Renwu_1'


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



CKPT_PATH="/home/ps/sdbench/models/sd_xl.safetensors"
LORA_PATH="/home/ps/sdbench/train/output/sdxl/Renwu/Renwu.safetensors;0.9"
GUIDANCE_SCALE=7.5
STEPS=50
N_SAMPLES=20
OUTPUT_PREFIX="generation/sdxl/sdxl_Renwu_1"
HEIGHT=1024
WIDTH=1024

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




# 设置常用的参数
CKPT_PATH="/home/ps/sdbench/models/1.ckpt"
OUTPUT_PREFIX="generation/sd/SD_Renwu_1"

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
        --images_per_prompt  20 \
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
        --network_weights "/home/ps/sdbench/models/lora/sdlora/chinese_painting_new.safetensors" \
        --network_mul 1
done

echo "All tasks completed."
