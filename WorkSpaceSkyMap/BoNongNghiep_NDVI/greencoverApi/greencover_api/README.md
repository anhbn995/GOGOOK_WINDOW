Green Cover
==================

Setting up
------------
### Setup environment
        conda env create -f geoai.yml
        conda activate geoai

### Download image from url
- Open download_url.py and change the parameters:

        month : list of month, ex: [1,2,3,5]
        year : interger number, ex: 2021
        start_date, end_date : interger number, end_date can be None if you want to get all of the month, ex: 1, None
        CLOUD_COVER : float number, ex :1.0 is 100% cloud cover
        name_ws : name of workspace on the eofactory.ai
        name_aoi : name of AOI on the eofactory.ai
        input_url : url of list workspace (recommend: don't change it)
        temp_path : path of the temporary folder, it can affect to PCI process
        folder_path : path of the destination folder
        token : you can get it on the eofactory.ai (notice: it is different for each user)

- After change it, you can download image:

        python download_url.py

### Execute the whole process
- The processes contains : crop image, cloud remove and greencover.
- You can run file with 2 options:
    - Open file and change the parameters if you set debug = True in the file.
    - Run direct on the terminal if you set debug = False in the file.

- If debug = True, you need change the parameters, below:

        static_result : True if you want to check result difference, bool
        temp_folder : path of the temporary folder, string
        cloud_tmp_dir : path of the folder ,it contains the results after run cloud remove and mosaic, string
        folder_paths : path of the input folder, string
        weight_path_cloud : path of the cloud weight, string
        weight_path_green : path of the green weight, string
        weight_path_water : path of the water weight, string

- After change the parameters, you can run it : 

        python run_v2.py

- If debug = False, you can run the file :

        python run_v2.py --folder_dir=[folder_paths] --temp_dir=[temp_folder] --cloud_tmp_dir=[cloud_tmp_dir] --weight_path_cloud=[weight_path_cloud] --weight_path_green=[weight_path_green] --weight_path_water=[weight_path_water] --check_result=[static_result]

### Only cloud remove
- If you want to run only it, you need to change the parameters in the file run_cloud_only.py, below:

        workspace : path of the input folder
        FN_MODEL : path of the cloud weight
        sort_amount_of_clouds : True if you want sort and choose the images by amount of the cloud areas
        first_image : path of the base image, can be None if you haven't any image

- After that, run file :

        python run_cloud_only.py

### Only green cover
- If you want to run only it, you need to change the parameters in the file run_classfication.py, below:

        input_path : path of the input folder
        weight_path_green : path of the green weight
        weight_path_water : path of the water weight
        dil : True if you want to dilate the green results 

- After that, run file :

        python run_classfication.py

