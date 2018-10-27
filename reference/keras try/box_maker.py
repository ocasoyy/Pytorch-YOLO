"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import os
import numpy as np
import pandas as pd
import glob

path = 'C:/Users/YY/Documents/Data/CCP/transf/*.txt'
def findFiles(path): return glob.glob(path)
filelist = findFiles(path)
# print(len(filelist))


def readLines(file):
    line = open(file, encoding='utf-8').read().strip()
    return line


text = []
for file in filelist:
    line = readLines(file)
    text.append(line)


outF = open("C:/Users/YY/Documents/boxes.txt", "w")
for line in text:
    # write line to output file
    outF.write(line)
    outF.write("/n")
outF.close()



