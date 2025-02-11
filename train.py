# https://mmtracking.readthedocs.io/en/latest/quick_run.html#run-with-customized-datasets-and-models
# https://github.com/open-mmlab/mmtracking/blob/master/demo/demo_mot_vis.py
# https://github.com/open-mmlab/mmtracking/tree/master/tools

import torch
import numpy as np

import argparse
import logging
import os
import os.path as osp
import uuid
from datetime import datetime
import random

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmtrack.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--unique_id', type=str,
                        default=uuid.uuid4().__str__())
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    OTP_DIR = os.environ.get("OTP_DIR", "/workspace/Final_Submission")
    # DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data/01_data")
    RANDOM_SEED = os.environ.get("RANDOM_SEED", 777)

    args = parse_args()

    # register all modules in mmtrack into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # cfg.data_root = DATA_DIR
    cfg.randomness = dict(seed=RANDOM_SEED, deterministic=True)
    cfg.gpu_ids = range(1)

    # cfg.custom_hooks = [
    #     dict(
    #         _scope_='mmdet',
    #         type='MMDetWandbHook',
    #         init_kwargs={
    #             'project': 'maicon-object-tracking',
    #             'name': f'{args.config}_{datetime.now().strftime("%m_%d-%H_%M_%S")}',
    #             'save_code': True,
    #             'group': 'object-tracking',
    #             'job_type': 'train',
    #             'resume': 'auto',
    #             'config': {
    #                 'dataset_type': cfg.dataset_type,
    #                 'data_root': cfg.data_root,
    #                 'train_pipeline': cfg.train_pipeline,
    #                 'test_pipeline': cfg.test_pipeline,
    #                 'train_dataloader': cfg.train_dataloader,
    #                 'val_dataloader': cfg.val_dataloader,
    #                 'test_dataloader': cfg.test_dataloader,
    #                 'model': cfg.model,
    #                 'train_cfg': cfg.train_cfg,
    #                 'val_cfg': cfg.val_cfg,
    #                 'test_cfg': cfg.test_cfg,
    #                 'param_scheduler': cfg.param_scheduler,
    #                 'optim_wrapper': cfg.optim_wrapper,
    #                 'randomness': cfg.randomness
    #             },
    #             'tags': [cfg.dataset_type, cfg.data_root, cfg.model.type, cfg.model.detector.backbone.type, cfg.train_cfg.max_epochs, cfg.optim_wrapper.optimizer.type],
    #             'id': args.unique_id
    #         },
    #         interval=10,
    #         log_checkpoint=True,
    #         log_checkpoint_metadata=True,
    #         num_eval_images=100,
    #         bbox_score_thr=0.3
    #     )
    # ]

    # Remove randomness
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                                '"auto_scale_lr.enable" or '
                                '"auto_scale_lr.base_batch_size" in your'
                                ' configuration file.')
    cfg.resume = args.resume
    # cfg.resume = True

    print(f'Config:\n{cfg.pretty_text}')

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
