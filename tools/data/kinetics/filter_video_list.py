# Copyright (c) Hikvision Research Institute. All rights reserved.
import os
import sys

from decord import VideoReader
from tqdm import tqdm

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} inp_list out_list')
        exit()

    inp_list, out_list = sys.argv[1:3]

    if 'train' in inp_list:
        folder = 'videos_train'
    elif 'val' in inp_list:
        folder = 'videos_val'
    else:
        folder = 'videos_test'
    video_dir = os.path.join(os.path.dirname(inp_list), folder)

    with open(inp_list) as f:
        full_list = f.readlines()

    with open(out_list, 'w') as f:
        for line in tqdm(full_list):
            video_name = line.split()[0]
            video_path = os.path.join(video_dir, video_name)
            try:
                vr = VideoReader(video_path, num_threads=1)
                assert len(vr) >= 1
                f.write(line)
            except Exception as e:
                print(str(e), video_path, os.path.getsize(video_path))
