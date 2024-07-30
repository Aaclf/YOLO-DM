from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo-dm.yaml")  # build a new model from scratch
    # YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8-DCNv4-FASFF.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # display model information
    model.info()

    # Use the model
    model.train(data="mydatasets-1.yaml", epochs=300, batch=8)  # train the model




