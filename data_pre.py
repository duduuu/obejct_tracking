# mmtracking/tools/dataset_converters


import click
import os, json, glob, shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

@click.group()
def cli():
    pass

AIHUB_TRACKING_CLASSES = [
    dict(id=0, name='person'),
    dict(id=1, name='person_on_vehicle'),
    dict(id=2, name='person_on_bike'),
    dict(id=3, name='person_on_kickboard'),
]

AIHUB_INCIDENT_CLASSES = [
    dict(id = 0, name = "bicycle_driving"),
    dict(id = 1, name = "bicycle_falldown"),
    dict(id = 2, name = "human_collision"),
    dict(id = 3, name = "human_falldown"),
    dict(id = 4, name = "human_falldown_bicycle_collision"),
    dict(id = 5, name = "human_falldown_kickboard_collision"),
    dict(id = 6, name = "human_following"),
    dict(id = 7, name = "human_invasion"),
    dict(id = 8, name = "human_loitering"),
    dict(id = 9, name = "human_motorbike"),
    dict(id = 10, name = "human_trooping"),
    dict(id = 11, name = "kickboard_bicycle_falldown"),
    dict(id = 12, name = "kickboard_driving"),
    dict(id = 13, name = "kickboard_falldown")
]


@cli.command()
@click.argument("aihub_dir")
def aihub2cocovid(aihub_dir):
    for file in os.listdir(aihub_dir):
        img = Image.open(os.path.join(dir, file))
        w, h = img.size
        
        if w == 750:
            print(file)


@cli.command()
@click.argument("dir")
def aihub2mot(dir):
    dir = "/data/val/label/2/"
    ann_dirs = glob.glob(dir + "*/*")
    det_list = []
    for ann_dir in ann_dirs:
        f2 = open(ann_dir + "/det.txt", "w")
        ann_file_dirs = sorted(glob.glob(ann_dir + "/*.json"))
        for ann_file in ann_file_dirs:
            with open(ann_file, "r", encoding="UTF-8") as f:
                json_data = json.load(f)

            frame_id = json_data["task"]["frame_id"] + 1
            for ann in json_data["annotation"]:
                track_id = ann["attributes"]["track_id"] # -1 for detections
                bbox_left = ann["bndbox"]["xmin"]
                bbox_top = ann["bndbox"]["ymin"]
                bbox_width = ann["bndbox"]["xmax"] - bbox_left
                bbox_height = ann["bndbox"]["ymax"] - bbox_top

                det = [frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, -1, -1, -1]
                det = ",".join(map(str, det)) + "\n"
                f2.write(det)
        f2.close()

cli.add_command(aihub2mot)
cli.add_command(aihub2cocovid)

if __name__ == "__main__":
    cli()