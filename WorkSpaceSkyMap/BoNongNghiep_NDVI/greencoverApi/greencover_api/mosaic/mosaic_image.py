import os
from re import A
import shutil
from glob import glob
from skyamqp import AMQP_Client

def _prepare_input_file(temporary_folder, file_list):
    input_text = f'{temporary_folder}/input.txt'
    input_content = ''

    for file_path in file_list:
        input_content += f'\"{file_path}\"\n'

    with open(input_text, 'w') as cursor:
        cursor.write(input_content)
    return input_text


def connect_to_client():
    connection = AMQP_Client(host='192.168.4.100',
                            port='5672',
                            virtual_host='/eof',
                            username='eof_rq_worker',
                            password='123',
                            heartbeat=5)
    print('**Connected AMQP host!!')
    return connection

def move_img(img_folder, out_path):
    list_img = glob(os.path.join(img_folder,'*.tif'))
    # print(img_folder)
    print("**List image mosaic:", list_img)
    for i in list_img:
        name_img = os.path.basename(i)
        in_folder = i
        out_folder = os.path.join(out_path, name_img)
        shutil.copyfile(in_folder, out_folder)

def check_base_img(base_img):
    if os.path.exists(base_img):
        try:
            img_base = glob(os.path.join(base_img,'*.tif'))[0]
        except:
            return False
        return img_base
    else:
        return False

def main(input_folder, cloud_tmp_dir, temp_folder, base_img, name):
    if os.path.exists(cloud_tmp_dir):
        pass
    else:
        os.mkdir(cloud_tmp_dir)

    if os.path.exists(temp_folder):
        pass
    else:
        os.mkdir(temp_folder)
    
    if base_img:
        img_base = check_base_img(base_img)
        if isinstance(img_base, bool):
            print("***Merge without base image***")
        else:
            print("***Merge with base image***")
            # cloud_tmp_dir_base = os.path.join(cloud_tmp_dir, os.path.basename(img_base))
            cloud_tmp_dir_base = os.path.join(cloud_tmp_dir, 'AAA_base.tif')
            shutil.copyfile(img_base, cloud_tmp_dir_base)
    else:
        print("***Merge without base image***")

    move_img(input_folder, cloud_tmp_dir)
    list_file_path_mosaic = []
    for path in glob(os.path.join(cloud_tmp_dir,'*.tif')):
        list_file_path_mosaic.append(path)
    list_file_path_mosaic = sorted(list_file_path_mosaic)
    # print(list_file_path_mosaic)
    input_file = _prepare_input_file(temp_folder, list_file_path_mosaic)
    pre_translate_path = f'{temp_folder}/%s.tif'%(name)

    connection = connect_to_client()
    gxl_rpcClient = connection.create_RPC_Client('gxl-python')
    response = gxl_rpcClient.send('mosaic', {
        "mfile": f'{temp_folder}/input.txt',
        "out_path": pre_translate_path
    })

    if not response['success']:
        raise Exception(response['message'])
    print("**Finished merge and mosaic")
    return pre_translate_path

if __name__ == '__main__':
    input_folder = '/home/nghipham/Desktop/Jupyter/data/DA/5_India/Hyderabad/tmp/T6_cut'
    cloud_tmp_dir = '/home/geoai/geoai_data_test/mosaic/T6'
    temp_folder = '/home/geoai/geoai_data_test/temp'
    name = 'T6'
    img_T1_PCI = '/home/nghipham/Desktop/Jupyter/data/DA/5_India/Hyderabad/results/T1/T1.tif'
    base_img = None

    main(input_folder, cloud_tmp_dir, temp_folder, base_img, name)