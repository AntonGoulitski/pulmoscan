import cv2
import os
import pydicom

inputdir = 'dcms/'
outdir = 'x-ray_png/'
os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list[:10]:
    ds = pydicom.read_file(inputdir + f)
    img = ds.pixel_array 
    cv2.imwrite(outdir + f.replace('.dcm','.png'),img)
