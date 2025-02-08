"""
Adapted from salesforce@LAVIS and Vision-CAIR@MiniGPT-4. Below is the original copyright:
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

import video_llama.tasks as tasks
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank, init_distributed_mode
from video_llama.common.logger import setup_logger
from video_llama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from video_llama.common.registry import registry
from video_llama.common.utils import now

# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


# Custom collate function
def custom_collate_fn(batch):
    # Debugging: Print the structure of the batch
    print("Batch structure:", batch)  # Add this line to inspect the batch structure

    # Adjust unpacking based on the actual structure of the items in the batch
    # Example: If each item is a dictionary, you might do:
    if isinstance(batch[0], dict):
        # Check the keys in the first item to understand the structure
        print("Keys in first item:", batch[0].keys())  # Debugging line to check keys

        # Adjust the keys based on your dataset's structure
        data = [item['image'] for item in batch]  # Assuming 'image' is the correct key
        labels = [item['labels'] for item in batch]  # Assuming 'labels' is the correct key
    else:
        data, labels = zip(*batch)  # Original unpacking logic

    # Pad the data if they are sequences
    data = pad_sequence(data, batch_first=True)  # Adjust based on your data structure
    return data, labels


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())
    

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    # Create DataLoader for training
    train_dataset = datasets['vad_instruct']['train']  # Adjust this based on your dataset structure
    dataloader = DataLoader(train_dataset, batch_size=cfg.run_cfg.batch_size_train, shuffle=True, num_workers=cfg.run_cfg.num_workers, collate_fn=custom_collate_fn)

    # datasets['webvid']['train'][0]
    # import pdb;pdb.set_trace()
    # datasets
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    # Enable mixed precision training
    scaler = amp.GradScaler()
    
    for data in dataloader:
        # Check if data is a tuple
        if isinstance(data, tuple):
            inputs, labels = data  # Adjust based on the actual structure of the tuple
            # Assuming the tuple contains inputs and labels directly
            model_inputs = {
                'image': inputs,  # Assuming inputs is the image tensor
                # Add any other necessary keys here if needed
            }
        else:
            inputs = data['image']  # Assuming 'image' is the correct key
            labels = data['labels']  # Assuming 'labels' is the correct key
            model_inputs = {
                'image': inputs,
                'text_input': data['text_input'],  # Assuming 'text_input' is also needed
            }
        
        # Enable mixed precision training
        with amp.autocast():
            outputs = model(model_inputs)  # Pass the dictionary to the model
            loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    runner.train()


if __name__ == "__main__":
    main()
