from ultralytics import YOLO

# 1. Load a pre-trained model
model = YOLO("yolo26n.pt")

# 2. Train on the built-in COCO8 sample dataset (downloads automatically)
results = model.train(data="coco8.yaml", epochs=10)