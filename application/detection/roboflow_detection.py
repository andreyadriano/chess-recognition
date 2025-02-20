import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from roboflow import Roboflow
from application.config import *

def load_roboflow_model():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_PROJECT_ID)
    model = project.version(ROBOFLOW_MODEL_VERSION).model
    model.confidence = OBJECT_DETECTION_MIN_CONFIDENCE
    model.overlap = OBJECT_DETECTION_MIN_OVERLAP
    return model

def roboflow_detect_objects(model, image):
    if model is not None:
        print(f"[debug] requested object detection to Roboflow server")
        return model.predict(image)
    else:
        raise ValueError("Error: Roboflow model is not loaded.")

def find_bounding_box_center(object):
    return object["x"] + object["width"]//2, object["y"] + object["height"]//2
