import os
import json
import urllib
from re import A
import uuid
import shutil
import base64
import datetime
import requests

from tqdm import tqdm
from shapely import geometry
from skyamqp import AMQP_Client
# from download_url import write_json_file
from download_crop.download_util import last_day_of_month, convert_l1c_to_l2a, make_nas_temp_folder_in_root_data_folder,\
                                        get_raw_image, process_sentinel2_pci, get_bands_for_gdal, gdal_merge_file, \
                                        clip_image, _translate, compute_bound, stretch_v2

os.environ['AMQP_HOST']='192.168.4.100'
os.environ['AMQP_PORT']='5672'
os.environ['AMQP_VHOST']='/eof'
os.environ['AMQP_USERNAME']='eof_rq_worker'
os.environ['AMQP_PASSWORD']='123'
connection = AMQP_Client(
  host=os.environ.get('AMQP_HOST'),
  port=os.environ.get('AMQP_PORT'),
  virtual_host=os.environ.get('AMQP_VHOST'),
  username=os.environ.get('AMQP_USERNAME'),
  password=os.environ.get('AMQP_PASSWORD'),
  heartbeat=5
)
print('Connected AMQP host!!')

def write_json_file(data, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return name

def get_list_workspace(input_url, token):
    list_workspace = []
    workspace = requests.get(input_url, headers={'Authorization': token})
    f = workspace.json()
    for i in f['data']:
        list_workspace.append([i['id'], i['name']])
    return list_workspace

def search_img_url_v2(AOI, query):
    list_id = []
    for n,i in enumerate(AOI):
        # import pdb
        # pdb.set_trace()
        item = geometry.Polygon(i['geometry']['coordinates'][0])
        url = f"https://finder.creodias.eu/resto/api/collections/Sentinel2/search.json?geometry={item}&" + urllib.parse.urlencode(query, doseq=False, safe='[,]', quote_via=urllib.parse.quote)
        response = requests.get(url.replace('%20', ''))

        if response.ok:
            find_list = response.json()
            _a = find_list['properties']['links'][0]['href']
            response = requests.get(_a)
            if response.ok:
#                 print(n , response.json()['properties']['totalResults'])
                for j in response.json()['features']:
                    name_img = j['properties']['title'].replace('.SAFE', '')
                    date_time = j['properties']['completionDate']
                    if [name_img, date_time] not in list_id:
                        list_id.append([name_img, date_time])
    return list_id

# def search_img_url(month, year, start_date, end_date, CLOUD_COVER, AOI):
#     FROM_DATE = '%sT00:00:00.000Z'%str(datetime.date(year, month, start_date))
#     if end_date == None:
#         print("All day")
#         TO_DATE = '%sT00:00:00.000Z'%str(last_day_of_month(datetime.date(year, month, start_date)))
#     else:
#         TO_DATE = '%sT00:00:00.000Z'%str(datetime.date(year, month, end_date))
#     IMAGE_TYPES = ['Sentinel2L1C']
#     GEOMETRY = AOI
#     PLANET_USER = 'edecfc3ba34f49d0b9d0c8e947487169'
#     PLANET_PASS = ''

#     planet_auth = f"{PLANET_USER}:{PLANET_PASS}"
#     planet_auth_encode = planet_auth.encode("ascii")

#     base64_bytes = base64.b64encode(planet_auth_encode)
#     base64_string = base64_bytes.decode("ascii")

#     print("Search senl1c image with date...")
#     response = requests.post('https://api.planet.com/data/v1/quick-search', headers={
#         'Authorization': 'Basic ' + base64_string
#         }, json={
#             'item_types': IMAGE_TYPES,
#             'filter': {
#                 'type': 'AndFilter',
#                 'config': [
#                     {
#                         'type': 'GeometryFilter',
#                         'field_name': 'geometry',
#                         'config': GEOMETRY,
#                     },
#                     {
#                         'type': 'DateRangeFilter',
#                         'field_name': 'acquired',
#                         'config': {
#                             'gte': FROM_DATE,
#                             'lte': TO_DATE,
#                         },
#                     },
#                     {
#                         'type': 'RangeFilter',
#                         'field_name': 'cloud_cover',
#                         'config': {
#                             'lte': CLOUD_COVER,
#                         },
#                     }
#                 ]
#             }
#         })

#     if not response.ok:
#         raise Exception("Get list image : %s"%(response))
#     list_img = response.json()
#     print("Done")
#     return list_img

def download_image_v2(image_id_l1c, out_path, level=2, code='10M', product='L2A', 
                        LICENSE_PCI=True, has_hazerem=False, geom = None):
    print("Prepare to download image...")
    attempts = 0
    file_id = uuid.uuid4()
    
    print("Get list id senl2a image...")
    # id_lic = image_id_l1c[0]
    # date_img = str(int((image_id_l1c[1]).split('-')[1]))
    # name_month = 'T%s'%(date_img)
    # response1 = convert_l1c_to_l2a([id_lic])
    # image_id = response1[0][product]

    img_id_date = image_id_l1c
    image_id = img_id_date[0]
    date_img = str(int((img_id_date[1]).split('-')[1]))
    name_month = 'T%s'%(date_img)
    if not os.path.exists(os.path.join(out_path, name_month)):
        os.mkdir(os.path.join(out_path, name_month))
    newfile = os.path.join(out_path, name_month, image_id+'.tif')
    if os.path.exists(newfile):
        pass
    else:
        if LICENSE_PCI:
            temp_dir = make_nas_temp_folder_in_root_data_folder()
        print("Done")

        print("Download senl2a image...")
        download_utils_rpcClient = connection.create_RPC_Client('download-image-utils')
        resp = download_utils_rpcClient.send('sentinel2.download_from_aws', {'id': image_id, 'level': level})
        if not resp['success']:
            resp = download_utils_rpcClient.send('sentinel2.download_sentinel_hub', {'id': image_id})
            if not resp['success']:
                raise Exception('Please download at Imagery Guru !')

        raw_image = get_raw_image(resp)
        resp_download = resp

        dir_path = '/home/geoai/geoai_data_test/awsdata/sen2/%s.SAFE/tmp'%(image_id)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        out_path = "%s/%s.tif" % (dir_path, file_id.hex)
        pre_out_path = "%s/%s.tif" % (temp_dir, file_id.hex)
        print("Finished download senl2a.")

        print("Try to run PCI...")
        if LICENSE_PCI:
            try:
                attempts = process_sentinel2_pci(attempts, temp_dir, product, code, resp_download['save_path'],
                                                 pre_out_path, has_hazerem)
                if attempts == 0 :
                    print("attempts: %s,"%str(attempts), "Finished run PCI.")
            except Exception as e:
                print(e)
                pass

        stretch_path = "%s/%s.json" % (dir_path, file_id.hex)    
        if attempts > 3 or not LICENSE_PCI:
            print("Can't run PCI, change to run gdal...")
            gdal_merge_file(resp_download['save_path'],pre_out_path, get_bands_for_gdal(code, product), level)
            print("Finished run gdal.")

        if geom:
            print("Clip image with AOI...")
            computed_geom = geom
            clip_image(geom, pre_out_path, str(file_id), dir_path, temp_dir)
            print("Finished clip image.")
        else:
            print("Translate cog...")
            _translate(pre_out_path, out_path)
            computed_geom = compute_bound(out_path)
            print("Done")

        print("Stretch image...")
        stretch_v2(out_path, stretch_path)
        print("Done")
        rename_output = out_path.replace(os.path.basename(out_path), image_id+'.tif')
        os.rename(out_path, rename_output)
        shutil.copyfile(rename_output, newfile)
        shutil.rmtree(temp_dir)
        shutil.rmtree('/home/geoai/geoai_data_test/awsdata/sen2/%s.SAFE'%(image_id))
        # print("Finished all.")
    return newfile

# def download_image(image_id_l1c, out_path, level=2, code='10M', product='L2A', 
#                     LICENSE_PCI=True, has_hazerem=False, geom = None):
#     print("Prepare to download image...")
#     attempts = 0
#     file_id = uuid.uuid4()
    
#     print("Convert senl1c id to senl2a id...")
#     id_lic = image_id_l1c[0]
#     date_img = str(int((image_id_l1c[1]).split('-')[1]))
#     name_month = 'T%s'%(date_img)
#     response1 = convert_l1c_to_l2a([id_lic])
#     image_id = response1[0][product]
#     if not os.path.exists(os.path.join(out_path, name_month)):
#         os.mkdir(os.path.join(out_path, name_month))
#     newfile = os.path.join(out_path, name_month, image_id+'.tif')
#     if os.path.exists(newfile):
#         pass
#     else:
#         if LICENSE_PCI:
#             temp_dir = make_nas_temp_folder_in_root_data_folder()
#         print("Done")

#         print("Download senl2a image...")
#         download_utils_rpcClient = connection.create_RPC_Client('download-image-utils')
#         resp = download_utils_rpcClient.send('sentinel2.download_from_aws', {'id': image_id, 'level': level})
#         if not resp['success']:
#             resp = download_utils_rpcClient.send('sentinel2.download_sentinel_hub', {'id': image_id})
#             if not resp['success']:
#                 raise Exception('Please download at Imagery Guru !')

#         raw_image = get_raw_image(resp)
#         resp_download = resp

#         dir_path = '/home/geoai/geoai_data_test/awsdata/sen2/%s.SAFE/tmp'%(image_id)
#         if not os.path.exists(dir_path):
#             os.mkdir(dir_path)
#         out_path = "%s/%s.tif" % (dir_path, file_id.hex)
#         pre_out_path = "%s/%s.tif" % (temp_dir, file_id.hex)
#         print("Finished download senl2a.")

#         print("Try to run PCI...")
#         if LICENSE_PCI:
#             try:
#                 attempts = process_sentinel2_pci(attempts, temp_dir, product, code, resp_download['save_path'],
#                                                  pre_out_path, has_hazerem)
#                 if attempts == 0 :
#                     print("attempts: %s,"%str(attempts), "Finished run PCI.")
#             except Exception as e:
#                 print(e)
#                 pass

#         stretch_path = "%s/%s.json" % (dir_path, file_id.hex)    
#         if attempts > 3 or not LICENSE_PCI:
#             print("Can't run PCI, change to run gdal...")
#             gdal_merge_file(resp_download['save_path'],pre_out_path, get_bands_for_gdal(code, product), level)
#             print("Finished run gdal.")

#         if geom:
#             print("Clip image with AOI...")
#             computed_geom = geom
#             clip_image(geom, pre_out_path, str(file_id), dir_path, temp_dir)
#             print("Finished clip image.")
#         else:
#             print("Translate cog...")
#             _translate(pre_out_path, out_path)
#             computed_geom = compute_bound(out_path)
#             print("Done")

#         print("Stretch image...")
#         stretch_v2(out_path, stretch_path)
#         print("Done")
#         rename_output = out_path.replace(os.path.basename(out_path), image_id+'.tif')
#         os.rename(out_path, rename_output)
#         shutil.copyfile(rename_output, newfile)
#         shutil.rmtree(temp_dir)
#         shutil.rmtree('/home/geoai/geoai_data_test/awsdata/sen2/%s.SAFE'%(image_id))
#         # print("Finished all.")
#     return newfile

def main(folder_path, name_ws, name_aoi, input_url, token, month, year, start_date, end_date, CLOUD_COVER, data, json_path):
    print("Create workspace...")
    correct_ws = False
    correct_aoi = False
    list_workspace = get_list_workspace(input_url, token)
    for i in list_workspace:
        if name_ws in i:
            correct_ws = True
            id_workspace = i[0]
            name_workspace = i[1]
            workspace_path = os.path.join(folder_path, name_workspace)
            if not os.path.exists(workspace_path):
                os.mkdir(workspace_path)
 
    if not CLOUD_COVER:
        cloudCover = [0, 100]
    else:
        cloudCover = [0, int(CLOUD_COVER*100)]

    if not correct_ws:
        raise Exception("Incorrect name workspace.")
    print("Done")

    print("Get AOI...")
    aoi_url = "https://api-aws.eofactory.ai/api/workspaces/%s/aois?region=sea"%(id_workspace)
    aoi_geo = requests.get(aoi_url, headers={'Authorization': token})
    AOI = aoi_geo.json()
    
    for j in AOI['data']:
        if name_aoi == j['name']:
            correct_aoi = True
            GEOMETRY = j['geom']['features'][0]['geometry']
            AOI_GEOMETRY = []
            for l in  j['geom']['features']:
                AOI_GEOMETRY.append(l)

    if not correct_aoi:
        raise Exception("Incorrect name AOI.")
    print("Done")
    # json_path = os.path.join(workspace_path, 'requirements.json')

    if os.path.exists(json_path):
        f = open(json_path)
        data = json.load(f)
    else:
        raise Exception("File json isn't exists")
    
    if len(data['AOI']) == 0 :
        data.update({'AOI':AOI_GEOMETRY})
    
    if data['workspace_path'] != workspace_path:
        data.update({'workspace_path':workspace_path})
    
    write_json_file(data, json_path)

    

    if type(month) == list:
        print("A")
        for i in month:
            try:
                print(data['list_image']['%s'%(i)])
            except:
                data['list_image'].update({'%s'%(i):[]})
                
            if len(data['list_image']) == 0:
                data['list_image'].update({'%s'%(i):[]})
                id_image = []
            else:
                # print(data['list_image']['%s'%(i)])
                # data['list_image'].update({'%s'%(i):[]})
                if data['list_image']['%s'%(i)] == None:
                    id_image = []
                else:
                    id_image = data['list_image']['%s'%(i)]

            print("Download image T%s..."%(str(i)))
            # list_img = search_img_url(i, year, start_date, end_date, CLOUD_COVER, AOI=GEOMETRY)
            if not end_date:
                end_date = int(str(last_day_of_month(datetime.date(year, i, start_date))).split('-')[-1])
            query =  {
                'status': 'all',
                'dataset': 'ESA-DATASET',
                'maxRecords': 1000,
                'sortParam': 'startDate',
                'sortOrder': 'descending',
                'startDate': str(datetime.date(year, i, start_date)),
                'completionDate': str(datetime.date(year, i, end_date)),
                'processingLevel': 'LEVEL2A',
                'cloudCover': cloudCover
            }
            list_img = search_img_url_v2(AOI=AOI_GEOMETRY, query=query)

            if len(list_img) == 0:
                raise Exception("Please enter a valid API key")

            for j in tqdm(list_img):
                image_id_l1c = j
                if image_id_l1c not in id_image:
                    newfile = download_image_v2(image_id_l1c, workspace_path)
                    id_image.append(os.path.basename(newfile))
                    print('\n')
                    data['list_image'].update({'%s'%(str(i)) : id_image})
                    write_json_file(data, json_path)
                
            print("Done")

    else:
        print("B")
        print("Download image T%s..."%(str(month)))
        if data['list_image'].update({'%s'%(str(month)) : []}) == None:
            id_image = []
        else:
            id_image = data['list_image'].update({'%s'%(str(month)) : []})

        # list_img = search_img_url(month, year, start_date, end_date, CLOUD_COVER, AOI=GEOMETRY)
        if not end_date:
            end_date = int(str(last_day_of_month(datetime.date(year, month, start_date))).split('-')[-1])
        query =  {
            'status': 'all',
            'dataset': 'ESA-DATASET',
            'maxRecords': 1000,
            'sortParam': 'startDate',
            'sortOrder': 'descending',
            'startDate': str(datetime.date(year, month, start_date)),
            'completionDate': str(datetime.date(year, month, end_date)),
            'processingLevel': 'LEVEL2A',
            'cloudCover': cloudCover
        }
        list_img = search_img_url_v2(AOI=GEOMETRY, query=query)
        for i in tqdm(list_img):
            image_id_l1c = i
            if image_id_l1c not in id_image:
                newfile = download_image_v2(image_id_l1c, workspace_path)
                id_image.append(os.path.basename(newfile))
                print('\n')
                data['list_image'].update({'%s'%(str(i)) : id_image})
                write_json_file(data, json_path)
        print("Done")

    return workspace_path
    
