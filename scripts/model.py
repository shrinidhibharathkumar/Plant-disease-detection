from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("models/best12n.pt")  # load an official model
    IMG_PATH = "C:\\Projects\\Plant-disease-detection\\datasets\\test\\images\\Bell-pepper-plant-JPG_jpg.rf.072ba7b994cf3600e558d28009ff6958.jpg"
    # Predict with the model
    # result = model.predict(IMG_PATH,device="cpu")
    # result[0].show()
    # model.info()
    model.cfg
