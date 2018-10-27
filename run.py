from yolo_config import *
from DarkNet import DarkNet


config = Config()

darkNet = DarkNet(config)
print(darkNet)