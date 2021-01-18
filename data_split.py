import cv2
import pandas as pd
import os
from glob import glob
import shutil


outdir = 'input1/Pneumonia_EQ/'


all_xray_df = pd.read_csv('archive/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('equalized_archive','equalized_images*',  '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
print(all_xray_df.sample(3))
i=0
for index in all_xray_df.loc[all_xray_df['Finding Labels'] == 'No finding']['path']:
    i+=1
    img = cv2.imread(index, 0)
    cv2.imshow('image',img)
    cv2.imwrite('NORMAL',img)
    shutil.copy(index, "input1/Pneumonia_EQ")
print(i)