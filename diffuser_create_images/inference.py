# inf_lora_mps_safe.py
import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file  # ✅ Safetensors support

# -------- CONFIG --------
MODEL_REPO = "runwayml/stable-diffusion-v1-5"
CACHE_PATH = "/Users/srijon/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
LORA_WEIGHTS_PATH = "lora-output/pytorch_lora_weights.safetensors"

DEVICE = "mps"  # "cpu" fallback if MPS still crashes
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
IMAGE_HEIGHT = 384  # smaller to reduce memory usage
IMAGE_WIDTH = 384

PROMPTS = [
    "A cute cartoon puppy in a magical forest",
    "A friendly dragon reading a book",
    "A spaceship landing on a colorful alien planet"
]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- LOAD TOKENIZER AND TEXT ENCODER --------
print("Loading tokenizer and text encoder...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# -------- LOAD VAE --------
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(MODEL_REPO, subfolder="vae", torch_dtype=torch.float32)

# -------- LOAD UNET --------
print("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(
    MODEL_REPO,
    subfolder="unet",
    torch_dtype=torch.float32
)

# -------- APPLY LoRA WEIGHTS --------
print(f"Loading LoRA weights from {LORA_WEIGHTS_PATH}...")
lora_state_dict = load_file(LORA_WEIGHTS_PATH, device=DEVICE)
unet.load_state_dict(lora_state_dict, strict=False)
print("LoRA weights loaded successfully!")

# -------- LOAD SCHEDULER --------
print("Loading scheduler...")
scheduler = PNDMScheduler.from_pretrained(MODEL_REPO, subfolder="scheduler")

# -------- CREATE PIPELINE --------
print("Creating pipeline...")
pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # skip NSFW safety
    feature_extractor=None
)

pipe = pipe.to(DEVICE)

# -------- GENERATE IMAGES --------
for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating image {i+1}/{len(PROMPTS)} for prompt: '{prompt}'")
    image = pipe(
        prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH
    ).images[0]

    filename = os.path.join(OUTPUT_DIR, f"generated_image_{i+1}.png")
    image.save(filename)
    print(f"Saved image to: {filename}")

print("\nAll images generated successfully!")
