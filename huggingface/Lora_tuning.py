import os
import re
import json
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    MobileViTImageProcessor,
    MobileViTForImageClassification,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput

# =========================
# 1️⃣ Parse filenames → label
# =========================
image_dir = "images"

data = []
for fname in sorted(os.listdir(image_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    match = re.search(r"value_(\d+)_(\d+)", fname)
    if not match:
        continue
    value = float(f"{match.group(1)}.{match.group(2)}")
    data.append({
        "image_path": os.path.abspath(os.path.join(image_dir, fname)),
        "label": value
    })

print(f"Loaded {len(data)} samples")

# =========================
# 2️⃣ Normalize labels to [0, 1]
# =========================
all_values = [d["label"] for d in data]
label_min = min(all_values)
label_max = max(all_values)

records = [
    {
        "image_path": d["image_path"],
        "label": (d["label"] - label_min) / (label_max - label_min)
    }
    for d in data
]

print(f"Label range: {label_min} → {label_max}")

# =========================
# 3️⃣ Dataset
# =========================
dataset = Dataset.from_list(records)
dataset = dataset.train_test_split(test_size=0.15, seed=42)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})
print(dataset)

# =========================
# 4️⃣ Processor + transform
# =========================
model_name = "apple/mobilevit-small"
processor = MobileViTImageProcessor.from_pretrained(model_name)

def transform(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    encoded = processor(images=images, return_tensors="pt")
    return {
        "pixel_values": encoded["pixel_values"],
        "labels": torch.tensor(batch["label"], dtype=torch.float32)
    }

dataset.set_transform(transform)

# =========================
# 5️⃣ Base model with regression head
# =========================
class MobileViTForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = MobileViTForImageClassification.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, 1)

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone.mobilevit(pixel_values)
        pooled = outputs.pooler_output
        prediction = self.backbone.classifier(pooled).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(prediction, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=prediction.unsqueeze(-1)
        )

base_model = MobileViTForRegression(model_name)

# =========================
# 6️⃣ Apply LoRA to the backbone
# LoRA targets the attention query/value projection layers
# inside MobileViT's transformer blocks
# =========================
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

lora_config = LoraConfig(
    r=16,                        # rank — higher = more capacity, more params
    lora_alpha=32,               # scaling factor (effective lr = alpha/r)
    lora_dropout=0.1,
    bias="none",
    target_modules=[             # which layers to inject LoRA into
        "query",
        "value",
    ],
    # LoRA for feature extraction, not a standard HF task type
    # so we use FEATURE_EXTRACTION
    task_type=TaskType.FEATURE_EXTRACTION,
)

# Apply LoRA to the inner HF backbone (not the wrapper)
# This freezes all base weights and only trains LoRA adapters
base_model.backbone = get_peft_model(base_model.backbone, lora_config)

print("\n📊 Parameter breakdown after LoRA:")
print_trainable_parameters(base_model)
print()

# The regression head is not covered by LoRA — make sure it stays trainable
for param in base_model.backbone.classifier.parameters():
    param.requires_grad = True

# =========================
# 7️⃣ Metrics
# =========================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze(-1)
    mae_normalized = np.mean(np.abs(predictions - labels))
    mae_real = mae_normalized * (label_max - label_min)
    return {
        "mae_normalized": round(float(mae_normalized), 4),
        "mae_real_units": round(float(mae_real), 3)
    }

# =========================
# 8️⃣ Training arguments
# LoRA trains far fewer params so we can afford higher lr
# =========================
training_args = TrainingArguments(
    output_dir="./mobilevit_lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    learning_rate=3e-4,          # higher lr is fine with LoRA
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mae_normalized",
    greater_is_better=False,
    remove_unused_columns=False,
    logging_steps=5,
    fp16=False,
)

# =========================
# 9️⃣ Trainer
# =========================
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

# =========================
# 🔟 Save
# LoRA saves only the adapter weights (tiny) separately from base model
# =========================
os.makedirs("./mobilevit_lora", exist_ok=True)

# Save LoRA adapter weights only (~small)
base_model.backbone.save_pretrained("./mobilevit_lora/lora_adapter")

# Save regression head separately
torch.save(
    base_model.backbone.classifier.state_dict(),
    "./mobilevit_lora/regression_head.pt"
)

# Save processor + label scale
processor.save_pretrained("./mobilevit_lora")
with open("./mobilevit_lora/label_scale.json", "w") as f:
    json.dump({"min": label_min, "max": label_max}, f)

print("✅ Done! Saved to ./mobilevit_lora")
print("   ├── lora_adapter/       ← LoRA weights only (small)")
print("   ├── regression_head.pt  ← regression linear layer")
print("   ├── preprocessor_config.json")
print("   └── label_scale.json")

# =========================
# 1️⃣1️⃣ Inference
# =========================
print("\n🔍 Running inference on first sample...")

base_model.eval()
sample = data[0]
image = Image.open(sample["image_path"]).convert("RGB")
encoded = processor(images=image, return_tensors="pt")

with torch.no_grad():
    out = base_model(pixel_values=encoded["pixel_values"])

pred_normalized = out.logits.squeeze().item()
pred_real = pred_normalized * (label_max - label_min) + label_min
print(f"True value : {sample['label']:.3f}")
print(f"Predicted  : {pred_real:.3f}")
