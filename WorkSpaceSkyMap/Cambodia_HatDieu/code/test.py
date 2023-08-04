# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:11:32 2022

@author: SkyMap
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:15:55 2022

@author: SkyMap
"""

from osgeo import gdal
import numpy as np
import pandas as pd
import geopandas as gp

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ReLU
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import load_model
    
import rasterio
np.random.seed()


def get_index_and_mask_train(fp_mask, nodata_value=0):
    src = rasterio.open(fp_mask)
    mask = src.read()[0].flatten()
    index_nodata = np.where(mask == nodata_value)
    mask_train = np.delete(mask, index_nodata)
    return mask_train, index_nodata


def get_df_flatten_train(fp_img, list_number_band, index_nodata):
    src = rasterio.open(fp_img)
    # return to img train
    list_band_have = list(range(1,src.count+1))
    dfObj = pd.DataFrame()
    if set(list_number_band).issubset(list_band_have):
        img = src.read(list_number_band)
        i = 0
        for band in img:
            band = band.flatten()
            band = np.delete(band, index_nodata)
            name_band = f"band {list_number_band[i]}"
            dfObj[name_band] = band
            i+=1
        return dfObj
    else:
        miss = np.setdiff1d(list_number_band, list_band_have)
        print("*"*15, "ERROR", "*"*15)
        print(f"Image dont have band : {miss.tolist()}")


fp_img = r"E:\WORK\Cambodia_HatDieu\img\Kampong Cham 11Mar2021 Mosaic.tif"
fp_mask = r"E:\WORK\Cambodia_HatDieu\label\Kampong Cham 11Mar2021 Mosaic_label.tif.tif"
list_band_to_train = [1,2,3,4]

mask_train, index_nodata = get_index_and_mask_train(fp_mask)
df = get_df_flatten_train(fp_img, list_band_to_train, index_nodata)
df['label'] = mask_train
g = df.groupby('label')
g = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    












