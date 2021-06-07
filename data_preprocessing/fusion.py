from nuscenes.nuscenes import NuScenes

data_dir = "D:/MULTIMEDIA/NuScenes"

nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)

all_samples = []

for scene in nusc.scene:
    first_sample_token = scene['first_sample_token']
    curr_sample = nusc.get('sample', first_sample_token)

    for _ in range(scene['nbr_samples']-1):
        all_samples.append(curr_sample)
        next_token = curr_sample['next']
        curr_sample = nusc.get('sample', next_token)
    all_samples.append(curr_sample)  # this appends the last sample of the scene

