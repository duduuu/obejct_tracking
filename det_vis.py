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
import glob
import numpy as np
from tqdm import tqdm
import cv2

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

    #img_dir = "/workspace/data/01_data"
    img_dir = "/workspace/data/01_data/train"
    gts_dir = "/workspace/ssw/baseline/YOLOX_outputs/yolox_x_track/track_submission"
    output_dir = gts_dir + "_vis"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #video_names = sorted(os.listdir(img_dir))
    video_names = sorted(glob.glob(gts_dir + "/*.txt"))
    for vname in video_names:
        video_name = osp.basename(vname)[:-4]
        # basic params
        # load video infos
        video_folder = osp.join(img_dir, video_name)
        # video-level infos
        img_folder = "img1"
        img_path = f'{video_folder}/{img_folder}'
        img_names = os.listdir(img_path)
        img_names = sorted(img_names)
        gts = mmengine.list_from_file(f'{gts_dir}/{video_name}.txt')
        img2gts = parse_gts(gts)
        
        if not os.path.exists(osp.join(output_dir, video_name)):
            os.makedirs(osp.join(output_dir, video_name))
        
        print(f"{video_name}...")
        for name in tqdm(img_names):
            img = cv2.imread(os.path.join(img_path, name))
            mot_frame_id = int(name.split('.')[0])
            gts = img2gts[mot_frame_id]
            for gt in gts:
                bbox = list(map(int,gt['bbox']))
                left, top, width, height = bbox
                for x in range(left, left+width):
                    for y in range(top, top+height):
                        img[y,x,0] = 200
            cv2.imwrite(osp.join(output_dir, video_name, name), img)
    

if __name__ == '__main__':
    main()