
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 modules/dev/flux_train.py \
  --pretrained_model_name_or_path "models/flux/flux1-dev.sft"  --clip_l "models/flux/clip_l.safetensors" --t5xxl "models/flux/t5xxl_fp16.safetensors" --ae "models/flux/ae.sft" \
  --save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 2 \
  --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
  --dataset_config "configs/datasets.toml" \
  --output_dir "/data/output/flux/Renwu_finetune" \
   --output_name "Renwu" \
  --learning_rate 5e-5 --max_train_epochs 6  --sdpa --highvram --cache_text_encoder_outputs_to_disk --cache_latents_to_disk --save_every_n_epochs 2 \
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
  --lr_scheduler constant_with_warmup --max_grad_norm 0.0 \
  --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
  --fused_backward_pass  --blocks_to_swap 8 --full_bf16 