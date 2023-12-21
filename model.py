import os
import random
import requests
from PIL import Image
from io import BytesIO

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path

# URL with host
LS_URL =  os.environ["LABEL_URL"]
LS_API_TOKEN = os.environ["LABEL_API"]


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                       'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                       'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                       'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                       'hair drier', 'toothbrush']

        # Load model
        self.model = YOLO("yolov8l.pt")

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = LS_URL + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN}
        
        # Getting URL and loading image
        image = Image.open(BytesIO(requests.get(
            full_url, headers=header).content))
        # Height and width of image
        original_width, original_height = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        
        # Getting prediction using model
        results = self.model.predict(image, imgsz=(original_height, original_width))       

        # Getting mask segments, boxes from model prediction
        for result in results:
            for i, box in enumerate(result.boxes):
                #XYXY
                x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]

                x = x_min * 100.0 / original_width
                y = y_min * 100.0 / original_height
                width = (x_max - x_min) * 100.0 / original_width
                height = (y_max - y_min) * 100.0 / original_height

                # Adding dict to prediction
                predictions.append({
                    "from_name" : self.from_name,
                    "to_name" : self.to_name,
                    "id": str(i),
                    "type": "rectanglelabels",
                    "score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "rotation": 0,
                        "rectanglelabels": [
                            self.labels[int(box.cls.item())]
                        ]
                    }})

                # Calculating score
                score += box.conf.item()


        print(10*"#", "Returned Prediction", 10*"#")

        # Dict with final dicts with predictions
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8s"
        }]

        return final_prediction
    
    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}