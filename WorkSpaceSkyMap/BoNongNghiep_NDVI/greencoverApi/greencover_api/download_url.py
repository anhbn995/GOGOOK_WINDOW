import os
import glob
import json
import shutil
from download_crop.download_from_url import main as download_image_from_url

def write_json_file(data, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return name

def get_weights(weight_path):
    all_weights = {}
    list_weights = glob.glob(os.path.join(weight_path, '*.h5'))
    for i in list_weights:
        if 'cloud' in os.path.basename(i):
            all_weights.update({'cloud': i})
        elif 'green' in os.path.basename(i):
            all_weights.update({'green': i})
        elif 'water' in os.path.basename(i):
            all_weights.update({'water': i})
        else:
            raise Exception("Name of weights contains name of class.")
    if len(all_weights.keys())!=3:
        raise Exception("Not enough file weight, please check %s"%(weight_path))
    return all_weights

def main(folder_path, name_ws, name_aoi, input_url, token, month, year, start_date, end_date, CLOUD_COVER, data, json_path):
    workspace_path = download_image_from_url(folder_path, name_ws, name_aoi, input_url, token, month, year, 
                                            start_date, end_date, CLOUD_COVER, data, json_path)
    return workspace_path

if __name__=="__main__":
    month = [12]
    year = 2021
    start_date = 1
    # end_date = None is download all 
    end_date = None
    CLOUD_COVER = 1.0
    # Check before run
    name_ws = "Test_download"
    # Check before run
    name_aoi = "MeaPhrik"
    input_url = 'https://api-aws.eofactory.ai/api/workspaces?region=sea'
    temp_path = '/home/geoai/geoai_data_test'
    weight_path = '/home/quyet/WorkSpace/Greencover_api/greencover/weights'
    folder_path = '/home/nghipham/Desktop/Jupyter/data/data_greencover_sing'
    # Check before run
    # token = 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzI5OCwibmFtZSI6IlF1eWV0IE5ndXllbiBOaHUiLCJlbWFpbCI6InF1eWV0Lm5uQGVvZmFjdG9yeS5haSIsImNvdW50cnkiOiJWaWV0bmFtIiwicGljdHVyZSI6bnVsbCwiaWF0IjoxNjM5OTkxMDc4LCJleHAiOjE2NDI1ODMwNzh9.O8xJbXle4ID2DzkIhdJbJOHiACw-gg2R-GWlyTaz60LeePCoDjCLVbVCDsd5KXo72H9lIMiw0UuaaQxDN8UQgQ'
    token ="Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTE3OSwibmFtZSI6IkR1YyBBbmggTmd1eWVuIiwiZW1haWwiOiJhbmguZG5AZW9mYWN0b3J5LmFpIiwiY291bnRyeSI6IlZpZXRuYW0iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDUuZ29vZ2xldXNlcmNvbnRlbnQuY29tLy02NXFhY2xaUTg2SS9BQUFBQUFBQUFBSS9BQUFBQUFBQUFBQS9BTVp1dWNrQUJpOWxmMmRST3Z5ZlctU2ZHZzF5dGFPMUNRL3Bob3RvLmpwZyIsImlhdCI6MTY0NDQ2MTYwOSwiZXhwIjoxNjQ3MDUzNjA5fQ.haNLK-v-Xgaw7YhcuItq8Jy6dyhvgkLW5EAv8-cjZAQGkOLpDx5DU3F6tbt9vQdREuNlp1PwSYkMTqsbmQZg2g"
    status_new = True
    name_json_file = os.path.join(os.getcwd(), 'requirements.json')
    if status_new:
        data = {'workspace_path' : "",
                'temp_path' : temp_path,
                'list_image' : {},
                'weights' : {},
                'AOI' : []}
        write_json_file(data, name_json_file)
    else:
        f = open(name_json_file)
        data = json.load(f)

    workspace_path = main(folder_path, name_ws, name_aoi, input_url, token, 
                            month, year, start_date, end_date, CLOUD_COVER, data, name_json_file)
                            
    all_weights = get_weights(weight_path)
    data.update({'weights':all_weights})
    folder_download = os.path.join(workspace_path, 'requirements.json')
    shutil.copyfile(folder_download, os.path.join(os.getcwd(), 'requirements.json'))
    print("Finished all")


