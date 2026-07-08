import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

# Replace with your Hugging Face repository
MODEL_NAME = "LavanyaManila22/swin-eurosat-classifier"

print("Loading model...")

processor = AutoImageProcessor.from_pretrained("LavanyaManila22/swin-eurosat-classifier")

model = AutoModelForImageClassification.from_pretrained("LavanyaManila22/swin-eurosat-classifier")

model.eval()

print("Model Loaded Successfully!")


def predict(image):

    image = image.convert("RGB")

    inputs = processor(
        image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    top5 = torch.topk(probabilities, 5)

    predictions = []

    for probability, index in zip(top5.values, top5.indices):

        predictions.append({

            "label": model.config.id2label[int(index)],

            "confidence": probability.item()

        })

    return predictions