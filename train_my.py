# https://mmtracking.readthedocs.io/en/latest/quick_run.html#run-with-customized-datasets-and-models
# https://github.com/open-mmlab/mmtracking/blob/master/demo/demo_mot_vis.py
# https://github.com/open-mmlab/mmtracking/tree/master/tools

import torch
import numpy as np
from datetime import datetime

import wandb

import os
import uuid
import os.path as osp
import random
import click

import mmengine
from mmengine.utils import mkdir_or_exist
from mmengine.runner import set_random_seed, Runner


@click.command()
@click.argument("config_file")
@click.option('-u', '--unique-id', 'unique_id', default=uuid.uuid4().__str__(), required=False)
def main(config_file, unique_id):
    OTP_DIR = os.environ.get("OTP_DIR", "/workspace/Final_Submission")
    DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data/01_data")
    RANDOM_SEED = os.environ.get("RANDOM_SEED", 777)

    # Read configure file and change some configs
    cfg = mmengine.Config.fromfile(config_file)
    cfg.data_root = DATA_DIR
    cfg.train_dataloader.dataset.data_root = DATA_DIR
    cfg.test_dataloader = cfg.test_cfg = cfg.test_evaluator = None
    cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
    cfg.visualizer.name = 'mot_visualizer'

    cfg.work_dir = f'{OTP_DIR}/exps'
    cfg.randomness = dict(seed=RANDOM_SEED, deterministic=True)
    cfg.gpu_ids = range(1)

    cfg.default_hooks.append(
        dict(
            type='MMDetWandbHook',
            init_kwargs={
                'project': 'maicon-object-tracking',
                'name' : f'{config_file}_{datetime.now().strftime("%m_%d-%H_%M_%S")}',
                'save_code' : True,
                'group' : 'object-tracking',
                'job_type' : 'train',
                'resume' : 'auto',
                'config' : {
                    'dataset_type' : cfg.dataset_type,
                    'data_root' : cfg.data_root,
                    'train_pipeline' : cfg.train_pipeline,
                    'test_pipeline' : cfg.test_pipeline,
                    'train_dataloader' : cfg.train_dataloader,
                    'val_dataloader' : cfg.val_dataloader,
                    'test_dataloader' : cfg.test_dataloader,
                    'model' : cfg.model,
                    'train_cfg' : cfg.train_cfg,
                    'val_cfg' : cfg.val_cfg,
                    'test_cfg' : cfg.test_cfg,
                    'param_scheduler' : cfg.param_scheduler,
                    'optim_wrapper' : cfg.optim_wrapper,
                    'randomness' : cfg.randomness
                },
                'tags': [cfg.dataset_type, cfg.data_root, cfg.model.type, cfg.model.backbone.type, cfg.train_cfg.max_epochs, cfg.optim_wrapper.optimizer.type],
                'id' : unique_id
            },
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            bbox_score_thr=0.3
        )
    )

    # Remove randomness
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    print(f'Config:\n{cfg.pretty_text}')

    # Train Model
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    runner=Runner.from_cfg(cfg)
    runner.train()