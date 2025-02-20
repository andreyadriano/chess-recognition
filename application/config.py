from dotenv import load_dotenv
import os

load_dotenv()

# .env variables

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT_ID = os.getenv("ROBOFLOW_PROJECT_ID")
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION")

# globals

SHOW_IMAGES = False

OBJECT_DETECTION_MIN_CONFIDENCE = 50 # 50% confidence
OBJECT_DETECTION_MIN_OVERLAP = 30 # 30% overlap