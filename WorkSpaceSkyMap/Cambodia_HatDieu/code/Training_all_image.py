# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:15:55 2022

@author: SkyMap
"""

# from email.mime import base
# from osgeo import gdal
import numpy as np
import pandas as pd
# import geopandas as gp
import os, glob

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ReLU
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from keras.models import load_model
    
import rasterio
from sklearn import datasets
np.random.seed()


def get_key(my_dict, val):
    for key, value in my_dict.items():
         if val in value:
             return key
    return "key doesn't exist"


def get_index_and_mask_train(fp_mask, nodata_value=0):
    src = rasterio.open(fp_mask)
    infor = src.tags()
    value_caschew = get_key(infor, 'Cashew')


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


def create_data_train_from_ones_img(fp_img, fp_mask, list_band_to_train):
    mask_train, index_nodata = get_index_and_mask_train(fp_mask)
    df_dataset = get_df_flatten_train(fp_img, list_band_to_train, index_nodata)
    df_dataset['label'] = mask_train


    g = df_dataset.groupby('label', group_keys=False)
    g = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    g = pd.DataFrame(g)
    size = 3000       # sample size
    replace = False  # with replacement
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
    a= g.groupby('label', as_index=False).apply(fn)
    # print(a)
    return pd.DataFrame(a)


# chu y list cua 2 thang nay phai cung ten
def create_data_train_all_img(list_fp_img, list_fp_mask, list_band_to_train, out_fp_csv_train):
    print(list_fp_img)
    dir_name_img = os.path.dirname(list_fp_img[0])
    list_df_all = []
    for fp_mask in list_fp_mask:
        base_name = os.path.basename(fp_mask)
        fp_img = os.path.join(dir_name_img, base_name)
        df_tmp = create_data_train_from_ones_img(fp_img, fp_mask, list_band_to_train)
        print(df_tmp)
        list_df_all.append(df_tmp)
        result = pd.concat(list_df_all)
        print(result)
    print(np.unique(result['label'].to_numpy()))
    result.to_csv(out_fp_csv_train)

def create_data_train(csv_training):
    datasets = pd.read_csv(csv_training).iloc[:, 1:]
    X = datasets.iloc[:, :-1]
    Y = datasets.iloc[:, -1]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    Y = np_utils.to_categorical(encoded_Y)
    return X,Y





def train(input, label, classes=7, epochs=100, batch_size=100, shuffle=True, model_path='model.h5'):
    assert classes>=2, 'number classese must be more than 1'
    model = Sequential()
    model.add(Dense(8, input_dim = 4))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(10, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(10, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(10, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(ReLU())

    # model.add(Dense(10, activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(ReLU())

    # model.add(Dense(10, activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    if classes==2:
        # model.add(Dense(2, activation = 'sigmoid'))
        loss = 'binary_crossentropy'
    else:
        # model.add(Dense(classes, activation = 'softmax'))
        loss='categorical_crossentropy'

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.fit(input, label, epochs=epochs, batch_size=batch_size, shuffle=shuffle)
    scores = model.evaluate(input, label)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save(model_path)

def main():
    list_fp_img = glob.glob(r'E:\WORK\Cambodia_HatDieu\Data\img\*.tif')
    list_fp_mask = glob.glob(r'E:\WORK\Cambodia_HatDieu\Data\mask\*.tif')
    list_band_to_train = [1,2,3,4]
    out_fp_csv_train = r'E:\WORK\Cambodia_HatDieu\Data\training.csv'
    model_path=r"E:\WORK\Cambodia_HatDieu\Data\model.h5"

    if not os.path.exists(out_fp_csv_train):
        create_data_train_all_img(list_fp_img, list_fp_mask, list_band_to_train, out_fp_csv_train)
    X, Y = create_data_train(out_fp_csv_train)
    train(X, Y, classes=7, epochs=1000, batch_size=100000, shuffle=True, model_path=model_path)









# list_fp_img = glob.glob(r'E:\WORK\Cambodia_HatDieu\Data\test\img\*.tif')
# list_fp_mask = glob.glob(r'E:\WORK\Cambodia_HatDieu\Data\test\mask\*.tif')
# list_band_to_train = [1,2,3,4]
# out_fp_csv_train = r'E:\WORK\Cambodia_HatDieu\Data\training1.csv'
# model_path=r"E:\WORK\Cambodia_HatDieu\Data\model.h5"

# if not os.path.exists(out_fp_csv_train):
#     create_data_train_all_img(list_fp_img, list_fp_mask, list_band_to_train, out_fp_csv_train)







    












