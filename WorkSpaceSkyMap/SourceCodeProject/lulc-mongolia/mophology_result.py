# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:32:05 2022

@author: AnhHo
"""

import rasterio
import cv2
import numpy as np
from matplotlib import pyplot as plt
with rasterio.open(r"Z:\Linh\stack_capital\Rs\Dense_v2_model_Dense_add2Dense_2022_04_20with17h04m40s_label_mask_nobuildup.tif") as src:
    predict = src.read(1)
    meta = src.meta.copy()
    mask_rest = (src.read_masks(1)).astype(np.uint8)
plt.figure()
plt.imshow(mask_rest)
plt.show()
#
class_value_sort = [1,6,3,2,7,5,4]
#mask_rest = (np.ones(predict.shape)*255).astype(np.uint8)
list_mask =[]
kernel = np.ones((3,3),np.uint8)
for value in class_value_sort[0:]:
    mask_class = ((predict == value)*255).astype(np.uint8)
    mask_class = cv2.bitwise_and(mask_rest, mask_class)
    mask_class = cv2.morphologyEx(mask_class, cv2.MORPH_CLOSE, kernel)
    mask_class = cv2.morphologyEx(mask_class, cv2.MORPH_OPEN, kernel)
    
    mask_class = cv2.bitwise_and(mask_rest, mask_class)
    mask_rest = mask_rest - mask_class
#    plt.figure()
#    plt.imshow(mask_class)
#    plt.show()
    list_mask.append(mask_class)
list_mask[-1]= cv2.bitwise_or(mask_class,mask_rest)
mophol_result = np.zeros(predict.shape).astype(np.uint8)
list_pixel = []
for i in range(len(class_value_sort)):
    value = class_value_sort[i]
    mask_i = list_mask[i]==255
#    print(value,np.count_nonzero(mask_i))
    list_pixel.append(np.count_nonzero(mask_i))
    
    mophol_result[mask_i] = value
#    plt.figure()
#    plt.imshow(mophol_result)
#    plt.show()
#
with rasterio.open(r"Z:\Linh\stack_capital\Rs\Dense_v2_model_Dense_add2Dense_2022_04_20with17h04m40s_label_mask_nobuildup_mol.tif",'w', **meta) as dst:
   dst.write(mophol_result, indexes=1)
#    
# total = sum(list_pixel)
# for i in range(len(class_value_sort)):
#     value = class_value_sort[i]
#     print(value,list_pixel[i],list_pixel[i]/total*100)
    