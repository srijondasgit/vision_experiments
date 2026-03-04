export PYTORCH_ENABLE_MPS_FALLBACK=1
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./train_images" \
  --output_dir="./lora-output" \
  --instance_prompt="a cartoon poem illustration" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --max_train_steps=800 \
  --mixed_precision="fp16"
