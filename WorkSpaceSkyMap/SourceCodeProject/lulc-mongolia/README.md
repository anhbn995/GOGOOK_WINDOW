1. Data train ở ổ **ml_data\DucAnh\WORK\Mongolia**
    - Img_same_size: Thư mục chứa ảnh LS8 và LS9.
    - Mask_label: chứa nhãn dưới dạng shapefile và raster.
    - Model_RiengBuildUp: Cần tạo một mô hình riêng về BuildUp để kết quả đk tốt.
    - Result: 
        - Buildiup_result.tif là kết quả của mô hình unet chạy ra BuildUp.
        - Dense_v2_model_Dense_add2Dense_2022_04_20with17h04m40s_label_mask_nobuildup_mol.tif là kết quả ver1.
        - ketquamau: là ket quả demo.
    - Model: chứa model mạng Dense 
2. Môi trường dùng: **geoai**
3. Run file Mongolia_TimeSeri.py
```
dir_img = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\aa"
fp_mask = r"E:\WORK\Mongodia\ThuDo_monggo\label_mask\label_mask_v2.tif"
list_number_band = [1,2,3,4,5,6,7]
out_fp_csv_train = r"E:\WORK\Mongodia\ThuDo_monggo\Data_training\dung_4anh.csv"
```

**dir_img** là dường dẫn **Img_same_size**\
**fp_mask** là dường dẫn **Mask_label**\
**list_number_band** là những band chọn ra để train\
**out_fp_csv_train** là đường dẫn chứa dữ liệu train đk lưu vào file csv

4. Run file Mongolia_TimeSeri_Predict.py được kết quả là 7 lớp nhưng k chứa lớp building.

5. Run file add_building_and_colormap.ipynb để thêm lớp building.

6. Run file mophology_result.py để có kết quả lm mịn tốt hơn.
