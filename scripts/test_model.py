# test_model.py
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

print("Downloading/loading model... this may take a few minutes the first time.")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

print("\nModel loaded successfully!")
print("Number of classes:", len(model.config.id2label))
print("Classes:", model.config.id2label)