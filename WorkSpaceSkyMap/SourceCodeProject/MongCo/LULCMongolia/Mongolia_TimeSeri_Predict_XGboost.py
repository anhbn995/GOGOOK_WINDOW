import xgboost as xgb

import os, glob
import numpy as np
import pandas as pd
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import rasterio
from rasterio.windows import Window
np.random.seed()


def write_window_many_chanel(output_ds, arr_c, s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop):
    for c, arr in enumerate(arr_c):
        output_ds.write(arr[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes= c + 1)


def get_list_image_by_time(dir_img):
    list_name_file = os.listdir(dir_img)
    print(list_name_file)
    list_time = []
    for name in list_name_file:
        list_time.append(name[17:25])
    list_time.sort(key=lambda date: datetime.strptime(date, '%Y%m%d'))
    return list_time[:9]



def get_df_flatten_predict(img_window, list_number_band, name_atrr):
    dfObj = pd.DataFrame()
    i = 0
    for band in img_window:
        band = band.flatten()
        name_band = f"band {list_number_band[i]}_{name_atrr}"
        dfObj[name_band] = band
        i+=1
    return dfObj


def create_df_from_window(list_fp_img_large, list_name_file_sort, list_number_band, w_crop_start, h_crop_start, size_w_crop, size_h_crop):
    fp_large_first = [s for s in list_fp_img_large if list_name_file_sort[0] in s][0]
    with rasterio.open(fp_large_first) as src:
            img_window = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
    df_all_predict = get_df_flatten_predict(img_window, list_number_band, list_name_file_sort[0])

    for name_file_sort in list_name_file_sort[1:]:
        fp_large = [s for s in list_fp_img_large if name_file_sort in s][0]
        with rasterio.open(fp_large) as src:
            img_window = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        df_one_img = get_df_flatten_predict(img_window, list_number_band, name_file_sort)
        df_all_predict = pd.concat([df_all_predict, df_one_img], axis=1)
    return df_all_predict, img_window.shape[1:]


def predict_df(df_all_predict, model, crop_size, shape_win):
    df_all_predict = xgb.DMatrix(df_all_predict)
    X_predict = model.predict(df_all_predict)
    X_predict = np.reshape(X_predict, (-1, shape_win[0], shape_win[1]))
    return X_predict.astype('uint8')
    


def predict_df2(df_all_predict, model, crop_size, shape_win):
    para = 4
    predictions = model.predict(df_all_predict, batch_size=1000, verbose=1)
    predictions = np.transpose(predictions)
    predictions = np.reshape(predictions, (-1, shape_win[0], shape_win[1]))
    print('shape of predictions', predictions.shape)
    return np.array([np.argmax(predictions, axis=0).astype('uint8')])


def predict_big(out_fp_predict, list_fp_img_large, list_name_file_sort, list_number_band, crop_size, model):
    fp_large_first = [s for s in list_fp_img_large if list_name_file_sort[0] in s][0]
    with rasterio.open(fp_large_first) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
        dtype_or = src.dtypes
        
    with rasterio.open(out_fp_predict, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, 
                                crs=source_crs,
                                transform=source_transform,
                                nodata=0,
                                dtype=rasterio.uint8,
                                compress='lzw') as output_ds:
        output_ds = np.empty((1,h,w))
        
        
    input_size = crop_size    
    padding = int((input_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))

    with rasterio.open(out_fp_predict,"r+") as output_ds:
        with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
            for start_h_org in list_hight:
                for start_w_org in list_weight:
                    h_crop_start = start_h_org - padding
                    w_crop_start = start_w_org - padding
                    if h_crop_start < 0 and w_crop_start < 0:
                        h_crop_start = 0
                        w_crop_start = 0
                        size_h_crop = crop_size + padding
                        size_w_crop = crop_size + padding
                        
                        df_all_predict, shape_win = create_df_from_window(list_fp_img_large, list_name_file_sort, list_number_band, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1
                        write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding, padding, crop_size + padding, 
                                                                        start_w_org, start_h_org, crop_size, crop_size)
                    elif h_crop_start < 0:
                        h_crop_start = 0
                        size_h_crop = crop_size + padding
                        size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
                        if size_w_crop == w - start_w_org + padding:
                            end_c_index_w =  size_w_crop
                        else:
                            end_c_index_w = crop_size + padding

                        df_all_predict, shape_win = create_df_from_window(list_fp_img_large, list_name_file_sort, list_number_band, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1
                        write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding ,padding, end_c_index_w, 
                                                    start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size)
                    elif w_crop_start < 0:
                        w_crop_start = 0
                        size_w_crop = crop_size + padding
                        size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
                        
                        if size_h_crop == h - start_h_org + padding:
                            end_c_index_h =  size_h_crop
                        else:
                            end_c_index_h = crop_size + padding

                        df_all_predict, shape_win = create_df_from_window(list_fp_img_large, list_name_file_sort, list_number_band, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1
                        write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, crop_size + padding, 
                                                    start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org))
                    else:
                        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
                        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
                        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
                            end_c_index_h = size_h_crop
                            end_c_index_w = size_w_crop
                            
                        elif size_w_crop < (crop_size + 2*padding):
                            end_c_index_h = crop_size + padding
                            end_c_index_w = size_w_crop
                            
                        elif size_h_crop < (crop_size + 2*padding):
                            end_c_index_w = crop_size + padding
                            end_c_index_h = size_h_crop
                            
                        else:
                            end_c_index_w = crop_size + padding
                            end_c_index_h = crop_size + padding
                            
                        df_all_predict, shape_win = create_df_from_window(list_fp_img_large, list_name_file_sort, list_number_band, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1 
                        write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                                                    start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
                    pbar.update()


out_fp_predict = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\predict\Xgboost.tif"
dir_img = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\Img_same_size"
list_number_band = [1,2,3,4,5,6,7]
model_fp = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\model\model_XGboost_2022_04_20with11h46m23s_train_0_7label.model"
crop_size = 1000

list_fp_img_large = glob.glob(os.path.join(dir_img, "*.tif"))
list_name_file_sort = get_list_image_by_time(dir_img)
print(list_name_file_sort)
model = xgb.Booster({'nthread': 20})
model.load_model(model_fp)
predict_big(out_fp_predict, list_fp_img_large, list_name_file_sort, list_number_band, crop_size, model)
print('done')
