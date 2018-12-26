from yolo_config import *
from DarkNet import DarkNet, darknet_test


config = Config()

darkNet = DarkNet(config)
print(darkNet)
darknet_test()
