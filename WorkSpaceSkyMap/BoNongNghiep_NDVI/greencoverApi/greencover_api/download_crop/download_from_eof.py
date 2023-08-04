import os
import urllib
import requests
import progressbar

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_from_eof(folderpath, input_url, token, list_month=None):
    workspace = requests.get('https://api-aws.eofactory.ai/api/workspaces?region=sea', headers={'Authorization': token})
    f1 = workspace.json()

    print('*Check and create workspace...')
    for i in f1['data']:
        if input_url.split('/')[-2] in i['id']:
            name_workspace = i['name']
        workspace_path = os.path.join(folderpath, name_workspace)
        if os.path.exists(workspace_path):
            pass
        else:
            os.mkdir(workspace_path)

    response = requests.get('https://api-aws.eofactory.ai/api/workspaces/309b7bea-83ae-47f8-8e94-78b9f9fad799/imageries?region=sea', headers={'Authorization': token})
    f = response.json()

    print('*Beginning download ...')
    if not list_month:
        for j in f['data']['images']:
            month = int(f['data']['images'][0]['acquired'].split('/')[1])
            month_path = os.path.join(workspace_path, 'T'+str(month))
            if os.path.exists(month_path):
                pass
            else:
                os.mkdir(month_path)
            link_image = j['download_url'].replace('https://aws', 'https://api2')
            img_name = j['name']
            img_path = os.path.join(month_path, img_name+'.tif')
            if os.path.exists(img_path):
                print("Exist file %s"%(img_name+'.tif'))
                pass
            else:
                print("Start download %s"%(img_name+'.tif'))
                urllib.request.urlretrieve(link_image, img_path, show_progress)
    else:
        for j in f['data']['images']:
            for k in list_month:
                month_path = os.path.join(workspace_path, 'T'+str(k))
                if os.path.exists(month_path):
                    pass
                else:
                    os.mkdir(month_path)
                    
            link_image = j['download_url'].replace('https://aws', 'https://api2')
            img_name = j['name']
            date_time = int(j['acquired'].split('/')[1])
            if date_time in list_month:
                img_path = os.path.join(month_path, img_name+'.tif')
                if os.path.exists(img_path):
                    print("Exist file %s"%(img_name+'.tif'))
                    pass
                else:
                    print("Start download %s"%(img_name+'.tif'))
                    urllib.request.urlretrieve(link_image, img_path, show_progress)

    return workspace_path