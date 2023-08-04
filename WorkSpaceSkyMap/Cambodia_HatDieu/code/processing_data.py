import rasterio
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dir_label = r"E:\WORK\Cambodia_HatDieu\label"
list_fp = glob.glob(os.path.join(dir_label, "*.tif"))

list_label = [0,1,2,3,4,5,6]
df = pd.DataFrame(columns = list_label)

for fp in list_fp:
    src = rasterio.open(fp)
    img =  src.read()[0]
    count = np.unique(img, return_counts=True)
    df_tmp = pd.DataFrame([count[1]], columns = count[0])
    df = df.add(df_tmp, fill_value=0)
df.loc[len(df)] = list_label
df = df.T
df.columns = ["Number", "Label"]
df = df.drop([0])
df.plot(x ='Label', y='Number', kind = 'bar')
plt.show()


# import rasterio

# fp_img = r"D:\_Solar_Panel\safesync-files\care_xong_xoa\18DEC22041458-S2AS_R04C2-059584419010_01_P004\GTIFF_1_196.tif"
# src = rasterio.open(fp_img)

# fp_1band = r"D:\_Solar_Panel\safesync-files\care_xong_xoa\predict_18DEC22041458-S2AS_R02C1-059584419010_01_P004\model_4band_128_adam_solar_panel\aoi_val\GTIFF_0_0.tif"
# src1band = rasterio.open(fp_1band)





