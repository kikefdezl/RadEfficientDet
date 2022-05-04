from nuscenes import NuScenes
import yaml
import os
from tqdm import tqdm
import pandas as pd

def main():
    print(os.getcwd())
    with open('../config.yaml') as yamlfile:
        cfg = yaml.safe_load(yamlfile)
    nuscenes_dir = cfg['nuscenes_dir']
    dataset_version = cfg['dataset_version']

    nusc = NuScenes(version=dataset_version, dataroot=nuscenes_dir, verbose=True)

    list_of_night_rain_imgs = []
    for scene in tqdm(nusc.scene):
        conditions = ['Night', 'night', 'Rain', 'rain', 'Fog', 'fog', 'Snow', 'snow']
        if any(condition in scene['description'] for condition in conditions):
            print(scene['description'])
            first_sample_token = scene['first_sample_token']
            curr_sample = nusc.get('sample', first_sample_token)

            for i in range(scene['nbr_samples']):
                cam_data = nusc.get('sample_data', curr_sample['data']['CAM_FRONT'])
                list_of_night_rain_imgs.append(os.path.basename(cam_data['filename']))
                cam_data = nusc.get('sample_data', curr_sample['data']['CAM_FRONT_RIGHT'])
                list_of_night_rain_imgs.append(os.path.basename(cam_data['filename']))
                cam_data = nusc.get('sample_data', curr_sample['data']['CAM_FRONT_LEFT'])
                list_of_night_rain_imgs.append(os.path.basename(cam_data['filename']))
                cam_data = nusc.get('sample_data', curr_sample['data']['CAM_BACK'])
                list_of_night_rain_imgs.append(os.path.basename(cam_data['filename']))
                cam_data = nusc.get('sample_data', curr_sample['data']['CAM_BACK_RIGHT'])
                list_of_night_rain_imgs.append(os.path.basename(cam_data['filename']))
                cam_data = nusc.get('sample_data', curr_sample['data']['CAM_BACK_LEFT'])
                list_of_night_rain_imgs.append(os.path.basename(cam_data['filename']))

                try:
                    next_token = curr_sample['next']
                    curr_sample = nusc.get('sample', next_token)
                except KeyError:
                    pass

    new_dataset_dir = cfg['dataset_save_dir']

    names = ['filename', 'x1', 'y1', 'x2', 'y2', 'class']
    all_val_data = pd.read_csv(os.path.join(new_dataset_dir, 'val.csv'), names=names)
    kept_rows = []
    for id, row in tqdm(all_val_data.iterrows()):
        if os.path.basename(row['filename']) in list_of_night_rain_imgs:
            kept_rows.append([row['filename'], row['x1'], row['y1'], row['x2'], row['y2'], row['class']])

    savepath = os.path.join(new_dataset_dir, 'val_night_rain.csv')
    dataframe = pd.DataFrame(kept_rows)
    dataframe.to_csv(savepath, header=False, index=False)

    print("Done.")




if __name__ == "__main__":
    main()