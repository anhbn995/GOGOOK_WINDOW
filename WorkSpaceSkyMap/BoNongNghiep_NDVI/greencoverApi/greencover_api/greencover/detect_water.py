import os
import cv2
import copy
import gdal
import rasterio
import numpy as np
import rasterio.mask
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from rasterio.windows import Window
from greencover.unet_models import models

def predict(image_path, result_path, weight_path, model, input_size_water):
    print("*Init water model")
    # model = models.unet_3plus((256, 256, 4), n_labels=1, filter_num=[32, 64, 128, 256, 512], 
    #                       filter_num_skip='auto', filter_num_aggregate='auto', stack_num_down=2, 
    #                       stack_num_up=1, activation='ReLU', output_activation='Sigmoid',batch_norm=True, 
    #                       pool=True, unpool=True, deep_supervision=True,  multi_input=True ,backbone='ResNet50', 
    #                       weights=None, freeze_backbone=False, freeze_batch_norm=True, name='unet3plus')
    model.load_weights(weight_path)
    id_image = os.path.basename(image_path).replace('.tif', '_water.tif')
    result_path = os.path.join(result_path, id_image)

    print("*Predict image")
    # num_band = input_size_water[-1]
    # input_size=input_size_water[0]
    # # num_band = 4
    # # input_size = 256
    # current_x, current_y=0,0
    # stride_size = input_size - 24
    # padding = int((input_size - stride_size) / 2)
    # with rasterio.open(image_path) as dataset_image:
    #     out_meta = dataset_image.meta
    #     imgs = dataset_image.read().swapaxes(0,1).swapaxes(1,2)
    #     # print(imgs.shape)

    #     # List h,w to stride window of image
    #     h,w = dataset_image.height, dataset_image.width
    #     img_1 = np.zeros((h,w))
    #     list_coordinates = []
    #     padding = int((input_size - stride_size) / 2)
    #     new_w = w + 2 * padding
    #     new_h = h + 2 * padding
    #     list_weight = list(range(padding, new_w - padding, stride_size))
    #     list_height = list(range(padding, new_h - padding, stride_size))
    #     with tqdm(total=len(list_height*len(list_weight))) as pbar:
    #         for i in range(len(list_height)):
    #             top_left_y = list_height[i]
    #             for j in range(len(list_weight)):
    #                 top_left_x = list_weight[j]
    #                 start_x = top_left_x - padding
    #                 end_x = min(top_left_x + stride_size + padding, new_w - padding)
    #                 start_y = top_left_y - padding
    #                 end_y = min(top_left_y + stride_size + padding, new_h - padding)
    #                 if start_x == 0:
    #                     x_off = start_x
    #                 else:
    #                     x_off = start_x - padding
    #                 if start_y == 0:
    #                     y_off = start_y
    #                 else:
    #                     y_off = start_y - padding
    #                 x_count = end_x - padding - x_off
    #                 y_count = end_y - padding - y_off
    #                 list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
    #                 image_detect = dataset_image.read(window=Window(x_off, y_off, x_count, y_count))[:num_band].swapaxes(0, 1).swapaxes(1, 2)
    #                 if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
    #                     img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
    #                     if start_x == 0 and start_y == 0:
    #                         img_temp[(input_size - image_detect.shape[0]):input_size, (input_size - image_detect.shape[1]):input_size] = image_detect
    #                     elif start_x == 0:
    #                         img_temp[0:image_detect.shape[0], (input_size - image_detect.shape[1]):input_size] = image_detect
    #                     elif start_y == 0:
    #                         img_temp[(input_size - image_detect.shape[0]):input_size, 0:image_detect.shape[1]] = image_detect
    #                     else:
    #                         img_temp[0:image_detect.shape[0], 0:image_detect.shape[1]] = image_detect
    #                     image_detect = img_temp
    #                 if np.count_nonzero(image_detect) > 0:
    #                     if len(np.unique(image_detect)) == 2 or len(np.unique(image_detect)) == 1:
    #                         y_pred = image_detect[:,:,0] 
    #                         pass
    #                     else:
    #                         image_detect = image_detect[np.newaxis,...]
    #                         y_pred = model.predict(image_detect)
    #                         if len(y_pred) >1:
    #                             y_pred = y_pred[-1]
    #                         y_pred = y_pred[0][:,:,0]
    #                         # y_pred = y_pred[0][:,:,0]
    #                 else:
    #                     y_pred = image_detect[:,:,0] 
    #                     pass
    #                 if start_x == 0 and start_y == 0:
    #                     y_pred = y_pred[padding:-padding,padding:-padding]
    #                 elif start_y == 0 and (x_count + x_off) < w:
    #                     y_pred = y_pred[padding:-padding,padding:-padding]
    #                 elif start_y == 0 and (x_count + x_off) >= w:
    #                     y_pred = y_pred[padding:-padding,padding:x_count]

    #                 elif (x_count + x_off) >= w and (y_count + y_off) < h:
    #                     y_pred = y_pred[padding:-padding,padding:x_count]
    #                 elif start_x == 0 and (y_count + y_off) < h:
    #                     y_pred = y_pred[padding:-padding:,padding:-padding]
    #                 elif start_x == 0 and (y_count + y_off) >= h:
    #                     y_pred = y_pred[padding:y_count:,padding:-padding]   

    #                 elif (x_count + x_off) >= w and (y_count + y_off) >= h:
    #                     y_pred = y_pred[padding:y_count,padding:x_count]
    #                 elif (y_count + y_off) >= h and (x_count + x_off) < w :
    #                     y_pred = y_pred[padding:y_count,padding:-padding]
    #                 else:
    #                     y_pred = y_pred[padding:x_count-padding:,padding:y_count-padding]
                    
    #                 # if y_pred.shape[1] <= int(stride_size/4) or y_pred.shape[0] <= int(stride_size/4):
    #                 if y_pred.shape[1] ==0:
    #                 # if y_pred.shape[1] <=11 or y_pred.shape[0] <=12:
    #                     pass
    #                 else:
    #                     if current_y >= w:
    #                         current_y = 0
    #                         current_x = current_x + past_i.shape[0]
                        
    #                     # if current_y+y_pred.shape[1] == w:
    #                     #     cuoi_y = True

    #                     # if cuoi_y and current_y ==0:
    #                     img_1[current_x:current_x+y_pred.shape[0],current_y:current_y+y_pred.shape[1]]+=y_pred
    #                     # else: 
    #                     #     pass
    #                     current_y += y_pred.shape[1]
    #                     past_i = y_pred
    #                 pbar.update()
    
    # img2 = copy.deepcopy(img_1)
    # img2[img2>0.85]=0
    # img2[img2!=0]=1

    # print("Write image...")
    # with rasterio.Env():
    #     profile = out_meta
    #     profile.update(
    #         dtype=rasterio.uint8,
    #         count=1,
    #         compress='lzw')
    # with rasterio.open(result_path, 'w', **profile) as dst:
    #     dst.write(img2.astype(np.uint8),1)
    # return img2

    num_band = 4
    input_size = 256
    INPUT_SIZE = 256
    crop_size = 100
    thresh_hold = 0.15
    thresh_hold = 1 - thresh_hold

    batch_size = 2
    dataset1 = gdal.Open(image_path)
    values = dataset1.ReadAsArray()[0:num_band]
    h,w = values.shape[1:3]    
    padding = int((input_size - crop_size)/2)
    padded_org_im = []
    cut_imgs = []
    new_w = w + 2*padding
    new_h = h + 2*padding
    cut_w = list(range(padding, new_w - padding, crop_size))
    cut_h = list(range(padding, new_h - padding, crop_size))

    list_hight = []
    list_weight = []
    for i in cut_h:
        if i < new_h - padding - crop_size:
            list_hight.append(i)
    list_hight.append(new_h-crop_size-padding)

    for i in cut_w:
        if i < new_w - crop_size - padding:
            list_weight.append(i)
    list_weight.append(new_w-crop_size-padding)

    img_coords = []
    for i in list_weight:
        for j in list_hight:
            img_coords.append([i, j])
    
    for i in range(num_band):
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    print(values.shape)
    del padded_org_im

    def get_im_by_coord(org_im, start_x, start_y,num_band):
        startx = start_x-padding
        endx = start_x+crop_size+padding
        starty = start_y - padding
        endy = start_y+crop_size+padding
        result=[]
        img = org_im[starty:endy, startx:endx]
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(num_band):
            result.append(cv2.resize(img[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC))
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            values, img_coords[i][0], img_coords[i][1],num_band)
        cut_imgs.append(im)

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in range(len(a)-1):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]])
        y_batch = model.predict(x_batch)
        y_batch = y_batch[-1]
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((INPUT_SIZE,INPUT_SIZE))
        true_mask = (true_mask>thresh_hold).astype(np.uint8)
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>thresh_hold).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]

    del cut_imgs
    mask_base = big_mask.astype(np.uint8)
    mask_base[mask_base==0]=2
    mask_base[mask_base==1]=0
    mask_base[mask_base==2]=1

    with rasterio.open(image_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
    new_dataset = rasterio.open(result_path, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(mask_base,1)
    new_dataset.close()
    return mask_base

