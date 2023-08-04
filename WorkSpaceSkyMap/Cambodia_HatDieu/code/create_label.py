import rasterio
import rasterio.mask
import glob
import numpy as np
import pandas as pd
import geopandas

geo_data = geopandas.read_file(r"X:\Linh\orkhon_mongolia\water.shp")
src1 = []
src2 = []
for i in range(len(geo_data)):
    if geo_data.iloc[i]['id'] == 1:
        src1.append(geo_data.iloc[i]['geometry'])
    else:
        src2.append(geo_data.iloc[i]['geometry'])

all_data= [[],[],[],[],[]]

for path in glob.glob('/mnt/Nam/bairac/data/xxx/*.tif'):
    with rasterio.open(path) as src:
        height = src.height
        width = src.width
        src_transform = src.transform
        data = src.read()
    mask_paddy = rasterio.features.geometry_mask(src1, (height, width), src_transform,invert=True, all_touched=True).astype(np.uint16)
    mask_background = rasterio.features.geometry_mask(src2, (height, width), src_transform,invert=True, all_touched=True).astype(np.uint16)
    mask = mask_paddy+2*mask_background
    ggg = np.concatenate((data, mask[np.newaxis,...]))
    data = []
    for i in range(ggg.shape[0]):
        all_data[i].extend(list(ggg[i][ggg[-1]>0]))

df = pd.DataFrame(np.array(all_data).T, columns = ['ban1', 'ban2', 'ban3', 'ban4', 'label'])
print(df)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('/mnt/Nam/tmp_Nam/pre-processing/data.csv')