
import numpy as np
import pandas as pd

from tqdm import tqdm
from keras.models import load_model

import rasterio
from rasterio.windows import Window
np.random.seed()


def write_window_many_chanel(output_ds, arr_c, s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop):
    for c, arr in enumerate(arr_c):
        output_ds.write(arr[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes= c + 1)


def get_df_flatten_predict(img_window, name_atrr):
    dfObj = pd.DataFrame()
    i = 0
    for band in img_window:
        band = band.flatten()
        name_band = f"band {i + 1}_{name_atrr}"
        dfObj[name_band] = band
        i+=1
    return dfObj


def create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop):
    with rasterio.open(fp_img_stack) as src:
        img_window = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
    df_all_predict = get_df_flatten_predict(img_window, 'band')  
    return df_all_predict, img_window.shape[1:]


def predict_df(df_all_predict, model, crop_size, shape_win):
    predictions = model.predict(df_all_predict, batch_size=1000, verbose=1)
    predictions = np.transpose(predictions)
    predictions = np.reshape(predictions, (-1, shape_win[0], shape_win[1]))
    print('shape of predictions', predictions.shape)
    return np.array([np.argmax(predictions, axis=0).astype('uint8')])


def predict_big(out_fp_predict, fp_img_stack, crop_size, model):
    with rasterio.open(fp_img_stack) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
        
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
                        
                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
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

                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
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

                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
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
                            
                        df_all_predict, shape_win =create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop)
                        img_predict = predict_df(df_all_predict, model, crop_size, shape_win) + 1 
                        write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                                                    start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
                    pbar.update()

def get_index_band_BF(name_img, list_fp):
    LC08_L2SP_131027_20210830_20210909_02_T1_0

out_fp_predict = r"Y:\DucAnh\WORK\Mongolia\predict\oke.tif"
fp_img_stack = r"X:\Linh\stack_capital\crop\stack_img.tif"
model_fp =r"Y:\DucAnh\WORK\Mongolia\model\model_Dense_add2Dense_2022_04_20with17h04m40s_label_mask_nobuildup.h5"
crop_size = 1000

model = load_model(model_fp)
predict_big(out_fp_predict, fp_img_stack, crop_size, model)
print('done')
