"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import glob
import os
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from time import time
from guided_diffusion.image_datasets import load_data_sample
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion import models
import datetime
import re
from torch.nn.functional import interpolate


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    starttime = time()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if args.use_dino:
        logger.log("Loading dino classification model")
        classification_model = models.get_dino_model()
    else:
        logger.log("Loading supervised classification model")
        classification_model = models.get_supervised_model()
    classification_model.eval()

    logger.log("Load data...")
    data = load_data_sample(
        data_dir=args.data_dir,
        batch_size=1,
    )

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model_savefiles = glob.glob(args.model_dir_path+'/model*')
    model_savefiles = natural_sort(model_savefiles)
    num_models=len(model_savefiles)

    logger.log("sampling...")
    all_images = []

    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    num_current_samples = 0

    while num_current_samples < args.num_samples:
        batch, _ = next(data)
        batch = batch[0:1].repeat(args.batch_size, 1, 1, 1).cuda()
        model_kwargs = {}

        with th.no_grad():
            feat = classification_model(batch).detach()
            model_kwargs["condition"] = feat

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        batch = interpolate(batch, size=32)
        batch = ((batch[0:1] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in batch])

        for savefile in model_savefiles:
            print(savefile.split('/')[-1])
            model.load_state_dict(
                dist_util.load_state_dict(savefile, map_location="cpu")
            )
            model.to(dist_util.dev())
            if args.use_fp16:
                model.convert_to_fp16()
            model.eval()

            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            samples = sample.contiguous()

            all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
            logger.log(f"created {len(all_images) * args.batch_size} samples")
        num_current_samples += 1

    arr = np.concatenate(all_images, axis=0)
    savefile = args.out_dir + args.model_dir_path.split('/')[-1] + '_' + args.name + '.jpeg'
    save_image(th.FloatTensor(arr).permute(0, 3, 1, 2), savefile, normalize=True,
               scale_each=True, nrow=args.batch_size*num_models + 1)
    print('Image saved at:', savefile)
    logger.log("sampling complete")
    print(time()-starttime)


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_dir_path="",
        out_dir="/home/deep-learning-advanced-course/results/sample_images",
        name='{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()),
        use_dino=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()