export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python modules/stable/sdxl_train_network.py \
    --network_module "networks.lora" \
    --pretrained_model_name_or_path "models/sd_xl.safetensors" \
    --dataset_config "configs/datasets.toml" \
    --output_dir "train/output" \
    --output_name "chinese_painting" \
    --save_model_as safetensors \
    --prior_loss_weight 1.0 \
    --max_train_epoch 50 \
    --learning_rate 1e-4 \
    --optimizer_type "AdamW8bit" \
    --xformers \
    --cache_latents \
    --gradient_checkpointing \
    --save_every_n_epochs 5 \