# Copyright (c) OpenMMLab. All rights reserved.
# This script converts MOT labels into COCO style.
# Official website of the MOT dataset: https://motchallenge.net/
#
# Label format of MOT dataset:
#   GTs:
#       <frame_id> # starts from 1 but COCO style starts from 0,
#       <instance_id>, <x1>, <y1>, <w>, <h>,
#       <conf> # conf is annotated as 0 if the object is ignored,
#       <class_id>, <visibility>
#

import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import mmcv
import numpy as np
from tqdm import tqdm
import cv2

# Classes in MOT:
CLASSES = [
    dict(id = 1, name = "pedestrian"),
]
"""
AIHUB_INCIDENT_CLASSES = [
    dict(id = 1, name = "bicycle_driving"),
    dict(id = 2, name = "bicycle_falldown"),
    dict(id = 3, name = "human_collision"),
    dict(id = 4, name = "human_falldown"),
    dict(id = 5, name = "human_falldown_bicycle_collision"),
    dict(id = 6, name = "human_falldown_kickboard_collision"),
    dict(id = 7, name = "human_following"),
    dict(id = 8, name = "human_invasion"),
    dict(id = 9, name = "human_loitering"),
    dict(id = 10, name = "human_motorbike"),
    dict(id = 11, name = "human_trooping"),
    dict(id = 12, name = "kickboard_bicycle_falldown"),
    dict(id = 13, name = "kickboard_driving"),
    dict(id = 14, name = "kickboard_falldown")
]
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT label and detections to COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    parser.add_argument(
        '--split-train',
        action='store_true',
        help='split the train set into half-train and half-validate.')
    return parser.parse_args()


def parse_gts(gts):
    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        conf = float(gt[6])
        category_id = int(gt[7])
        visibility = float(gt[8])
        anns = dict(
            category_id=category_id,
            bbox=bbox,
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf)
        outputs[frame_id].append(anns)
    return outputs


def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output)

    sets = ['train', 'test']
    if args.split_train:
        sets += ['half-train', 'half-val']
    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')
        if 'half' in subset:
            in_folder = osp.join(args.input, 'train')
        else:
            in_folder = osp.join(args.input, subset)
        out_file = osp.join(args.output, f'{subset}_cocoformat.json')
        outputs = defaultdict(list)
        outputs['categories'] = CLASSES
        video_names = os.listdir(in_folder)
        for video_name in tqdm(video_names):
            # basic params
            parse_gt = 'test' not in subset
            ins_maps = dict()
            # load video infos
            video_folder = osp.join(in_folder, video_name)
            # video-level infos
            img_folder = "img1"
            img_path = f'{video_folder}/{img_folder}'
            img_names = os.listdir(img_path)
            img_names = sorted(img_names)
            img = cv2.imread(os.path.join(img_path, img_names[0]))
            fps = 30
            num_imgs = len(img_names)
            height, width = img.shape[:2]
            video = dict(
                id=vid_id,
                name=video_name,
                fps=fps,
                width=width,
                height=height)
            # parse annotations
            if parse_gt:
                gts = mmengine.list_from_file(f'{video_folder}/gt/gt.txt')
                img2gts = parse_gts(gts)
            # make half sets
            if 'half' in subset:
                split_frame = num_imgs // 10 + 1
                if 'train' in subset:
                    img_names = img_names[split_frame:]
                elif 'val' in subset:
                    img_names = img_names[:split_frame]
                else:
                    raise ValueError(
                        'subset must be named with `train` or `val`')
                mot_frame_ids = [str(int(_.split('.')[0])) for _ in img_names]
                with open(f'{video_folder}/gt/gt_{subset}.txt', 'wt') as f:
                    for gt in gts:
                        if gt.split(',')[0] in mot_frame_ids:
                            f.writelines(f'{gt}\n')
            # image and box level infos
            for frame_id, name in enumerate(img_names):
                img_name = osp.join(video_name, img_folder, name)
                mot_frame_id = int(name.split('.')[0])
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_id=mot_frame_id)
                if parse_gt:
                    gts = img2gts[mot_frame_id]
                    for gt in gts:
                        gt.update(id=ann_id, image_id=img_id)
                        mot_ins_id = gt['mot_instance_id']
                        if mot_ins_id in ins_maps:
                            gt['instance_id'] = ins_maps[mot_ins_id]
                        else:
                            gt['instance_id'] = ins_id
                            ins_maps[mot_ins_id] = ins_id
                            ins_id += 1
                        outputs['annotations'].append(gt)
                        ann_id += 1
                outputs['images'].append(image)
                img_id += 1
            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = ins_id
        print(f'{subset} has {ins_id} instances.')
        mmengine.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    main()
    
# python ./mot2coco.py -i /workspace/data/01_data_ir/ -o /workspace/data/01_data_ir/annotations2 --split-train