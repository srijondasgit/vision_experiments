# Download a dataset from HF before trying this script

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
import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput

# =========================
# 1️⃣ Parse filenames → label
# id_1_value_359_439.jpg → 359.439
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
print("Sample values:", [d["label"] for d in data[:5]])

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
# 3️⃣ Dataset — store paths, not images
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
# set_transform always passes a BATCH (dict of lists), not a single example
# =========================
model_name = "apple/mobilevit-small"
processor = MobileViTImageProcessor.from_pretrained(model_name)

def transform(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    encoded = processor(images=images, return_tensors="pt")
    return {
        "pixel_values": encoded["pixel_values"],  # [B, C, H, W]
        "labels": torch.tensor(batch["label"], dtype=torch.float32)
    }

dataset.set_transform(transform)

# =========================
# 5️⃣ Regression model
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
        pooled = outputs.pooler_output           # [B, 640]
        prediction = self.backbone.classifier(pooled).squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(prediction, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=prediction.unsqueeze(-1)
        )

model = MobileViTForRegression(model_name)

# =========================
# 6️⃣ Metrics
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
# 7️⃣ Training arguments
# =========================
training_args = TrainingArguments(
    output_dir="./mobilevit_regression",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    learning_rate=1e-4,
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
# 8️⃣ Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

# =========================
# 9️⃣ Save
# =========================
os.makedirs("./mobilevit_regression", exist_ok=True)
torch.save(model.state_dict(), "./mobilevit_regression/model_weights.pt")
processor.save_pretrained("./mobilevit_regression")

with open("./mobilevit_regression/label_scale.json", "w") as f:
    json.dump({"min": label_min, "max": label_max}, f)

print("✅ Done! Model saved to ./mobilevit_regression")

# =========================
# 🔟 Quick inference test
# =========================
model.eval()
sample = data[0]
image = Image.open(sample["image_path"]).convert("RGB")
encoded = processor(images=image, return_tensors="pt")
with torch.no_grad():
    out = model(pixel_values=encoded["pixel_values"])
pred_normalized = out.logits.squeeze().item()
pred_real = pred_normalized * (label_max - label_min) + label_min
print(f"True value : {sample['label']:.3f}")
print(f"Predicted  : {pred_real:.3f}")
