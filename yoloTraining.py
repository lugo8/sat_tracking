from ultralytics import YOLO

def train(epoch):
    # Heavily inspired by code from https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/blob/master/local_env/train.py
    # Trains an object detection model with a specified number of epochs
    # using data specified by config.yaml
    #
    # Parameters:
    # - epoch: specified number of epochs
    
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    results = model.train(data="config.yaml", epochs=epoch, batch=4, imgsz=512, save=True, visualize=True, show=True, val=False)  # train the model

    return results
    
def val(path):   
    # Heavily inspired by code from https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/blob/master/local_env/train.py
    # Validates object detection model
    #
    # Parameters:
    # - epoch: path to desired model

    model = YOLO(path)  # load a custom model

    # Validate the model
    metrics = model.val(plots=True)  # no arguments needed, dataset and settings remembered
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category
    #print(metrics.box.map50)

    return metrics

if __name__ == "__main__":
    #train(15)
    val("runs/detect/train24/weights/best.pt")