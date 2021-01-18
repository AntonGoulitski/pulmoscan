import numpy as np
import cv2
import os


inputdir = 'archive/images/'
outdir = 'equalized_images'
os.mkdir(outdir)
test_list = [f for f in os.listdir(inputdir)]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for f in test_list:
    img = cv2.imread(inputdir + f, 0)
    cl1 = clahe.apply(img)
    cv2.imwrite(outdir + f, cl1)
