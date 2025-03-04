from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo12n.pt")  # load an official model

# Predict with the model
results = model.train(data="dataset/data.yaml", epochs=10, imgsz=640)

# results[0].show()