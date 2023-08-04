import os, glob
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

import xgboost as xgb    


def get_index_and_mask_train(fp_mask, nodata_value=0):
    src = rasterio.open(fp_mask)
    mask = src.read()[0].flatten()
    index_nodata = np.where(mask == nodata_value)
    mask_train = np.delete(mask, index_nodata)
    return mask_train, index_nodata


def get_df_flatten_train(fp_img, list_number_band, index_nodata, name_atrr):
    src = rasterio.open(fp_img)
    # print(src.height, src.width,"aaaaaa", fp_img)
    # return to img train
    list_band_have = list(range(1,src.count+1))
    dfObj = pd.DataFrame()
    if set(list_number_band).issubset(list_band_have):
        img = src.read(list_number_band)
        i = 0
        for band in img:
            band = band.flatten()
            band = np.delete(band, index_nodata)
            name_band = f"band {list_number_band[i]}_{name_atrr}"
            dfObj[name_band] = band
            i+=1
        return dfObj
    else:
        miss = np.setdiff1d(list_number_band, list_band_have)
        print("*"*15, "ERROR", "*"*15)
        print(f"Image dont have band : {miss.tolist()}")


def get_list_image_by_time(dir_img):
    list_name_file = os.listdir(dir_img)
    print(list_name_file)
    list_time = []
    for name in list_name_file:
        list_time.append(name[17:25])
    list_time.sort(key=lambda date: datetime.strptime(date, '%Y%m%d'))
    return list_time[:9]


def create_csv_train(list_fp_img, list_name_file_sort, list_number_band, index_nodata,fp_csv):
    fp_img_first = [s for s in list_fp_img if list_name_file_sort[0] in s][0]
    df = get_df_flatten_train(fp_img_first, list_number_band, index_nodata, list_name_file_sort[0])
    print(df.shape,"a")
    for name_file_sort in list_name_file_sort[1:]:
        fp_img = [s for s in list_fp_img if name_file_sort in s][0]
        df1 = get_df_flatten_train(fp_img, list_number_band, index_nodata, name_file_sort)
        df = pd.concat([df, df1], axis=1)
        print(df.shape)
    df.to_csv(fp_csv)    


def dao_ngay_df(df, list_colums):
    df_get = pd.concat([df.pop(x) for x in list_colums], axis=1)
    return pd.concat([df,df_get ], axis=1)


def make_time_seris(fp_csv):
    datasets = pd.read_csv(fp_csv)
    list_name_band = datasets.columns.to_list()
    tmp_df = datasets.copy()
    for i in range(8):
        list_7_band = list_name_band[0:7]
        tmp_df = dao_ngay_df(tmp_df, list_7_band)
        tmp_df.columns = list_name_band
        datasets = pd.concat([datasets, tmp_df])
    print(datasets.shape)
    return datasets


def create_data_train_Xgboost(df, training_per = 0.8):
    datasets = df.iloc[:, 2:]
    print(datasets)
    ds_train = datasets.sample(frac=training_per)
    ds_test = datasets[~datasets.isin(ds_train)].dropna()

    X_train = ds_train.iloc[:, :-1]
    Y_train = ds_train.iloc[:, -1]
    X_test = ds_test.iloc[:, :-1]
    Y_test = ds_test.iloc[:, -1]
    return X_train, Y_train, X_test, Y_test




name_time = datetime.now().strftime('%Y_%m_%dwith%Hh%Mm%Ss')
dir_img = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\Img_same_size"
fp_mask = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\Mask_label\label_mask_0.tif"
list_number_band = [1,2,3,4,5,6,7]
out_fp_csv_train = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\train_0_7label.csv"
name_training = os.path.basename(out_fp_csv_train)[:-4]
fp_model_save = f"E:\WORK\Mongodia\ThuDo_monggo\Data_training\model\model_XGboost_{name_time}_{name_training}.model"


list_name_file_sort = get_list_image_by_time(dir_img)
list_fp_img = glob.glob(os.path.join(dir_img, "*.tif"))

mask_train, index_nodata = get_index_and_mask_train(fp_mask, nodata_value=0)
if not os.path.exists(out_fp_csv_train):
    create_csv_train(list_fp_img, list_name_file_sort, list_number_band, index_nodata,out_fp_csv_train)

df = make_time_seris(out_fp_csv_train)
mask_train = np.tile(mask_train, 9)-1
print(np.unique(mask_train))
df['label'] = mask_train
df = df.reset_index()


X_train, Y_train, X_test, Y_test = create_data_train_Xgboost(df, training_per = 0.8)
print(X_test)
print('Training ...')
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)
num_round = 100
param = {'max_depth': 7, 'eta': 1, 'objective': 'multi:softmax'}
param['nthread'] = 12
param['eval_metric'] = 'auc'
param['num_class'] = 8
param['gpu_id'] = 0
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist)
bst.save_model(fp_model_save)