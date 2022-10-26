# Copyright (c) Hikvision Research Institute. All rights reserved.
# Generate video list for Mini-Kinetics-200
# https://github.com/s9xie/Mini-Kinetics-200

import os
import sys

import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} FULL_LIST YTID_LIST OUT_LIST\n'
              f'Example: {sys.argv[0]} kinetics400_train_list_videos.txt'
              'train_ytid_list.txt mini_kinetics200_train_list_videos.txt')
        exit()

    full_list_file, ytid_list_file, output_list_file = sys.argv[1:4]

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    input_label_map_file = os.path.join(cur_dir, 'label_map_k400.txt')
    output_label_map_file = os.path.join(cur_dir, 'label_map_mini_k200.txt')
    with open(input_label_map_file) as f:
        class_labels = [line.strip() for line in f]

    full_list = {}
    with open(full_list_file) as f:
        for line in f:
            line = line.strip()
            fields = line.split()
            assert len(fields) == 2
            filename = fields[0]
            class_id = int(fields[1])
            ytid = filename[:11]
            full_list[ytid] = (filename, class_id)

    class_ids = []
    out_list = []
    with open(ytid_list_file) as f:
        for line in f:
            ytid = line.strip()
            if ytid not in full_list:
                print(f'{ytid}: missing video')
                continue
            filename, class_id = full_list[ytid]
            class_ids.append(class_id)
            out_list.append(full_list[ytid])

    sorted_class_ids = np.sort(np.unique(class_ids))
    class_id_map = {x: i for i, x in enumerate(sorted_class_ids)}

    with open(output_list_file, 'w') as f:
        for filename, class_id in out_list:
            f.write(f'{filename} {class_id_map[class_id]}\n')

    with open(output_label_map_file, 'w') as f:
        for i in sorted_class_ids:
            f.write(f'{class_labels[i]}\n')
