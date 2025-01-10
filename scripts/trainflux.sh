export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 modules/dev/flux_train_network.py \
  --pretrained_model_name_or_path "models/flux/flux1-dev.sft" \
  --clip_l "models/flux/clip_l.safetensors" \
  --t5xxl "models/flux/t5xxl_fp16.safetensors" \
  --ae "models/flux/ae.sft" \
  --cache_latents_to_disk \
  --save_model_as safetensors \
  --sdpa --persistent_data_loader_workers \
  --max_data_loader_n_workers 2 \
  --seed 42 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --network_module networks.lora_flux \
  --network_dim 4 \
  --optimizer_type adamw8bit \
  --sample_prompts="chinese painting" \
  --sample_every_n_steps="1000" \
  --learning_rate 8e-4 \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --fp8_base \
  --highvram \
  --max_train_epochs 50 \
  --save_every_n_epochs 10 \
  --dataset_config "configs/datasets.toml" \
  --output_dir "train/output/flux/chinese_painting_1_7" \
  --output_name "chinese_painting" \
  --timestep_sampling shift \
  --discrete_flow_shift 3.1582 \
  --model_prediction_type raw \
  --guidance_scale 1 \
  --loss_type l2 \
  --logging_dir "train/logs" \