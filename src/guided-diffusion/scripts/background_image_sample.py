"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from time import time
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
from torch.nn.functional import interpolate
from PIL import Image


def load_image(path, top=None, left=None, height=None, width=None):
    img = Image.open('/home/deep-learning-advanced-course/data/val_50classes/' + path)
    zoomed_img = img
    img = transforms.functional.resize(img, [32, 32])
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    if top is not None:
        zoomed_img = transforms.functional.resized_crop(zoomed_img, top, left, height, width, [32, 32])
    else:
        zoomed_img = transforms.functional.resize(zoomed_img, [32, 32])
    zoomed_img = transforms.ToTensor()(zoomed_img)
    zoomed_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(zoomed_img)
    return img, zoomed_img


def main():
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
    image_paths = ['n03594945/ILSVRC2012_val_50classes_00005302.JPEG', 'n07749582/ILSVRC2012_val_50classes_00035888.JPEG',
                   'n02058221/ILSVRC2012_val_50classes_00048348.JPEG', 'n02058221/ILSVRC2012_val_50classes_00037746.JPEG',]
    tops = [30, 130, None, None]
    lefts = [140, 270, None, None]
    heights = [210, 165, None, None]
    widths = [350, 185, None, None]

    images = []
    for i in range(len(image_paths)):
        img, zoomed_img = load_image(image_paths[i], tops[i], lefts[i], heights[i], widths[i])
        images.append(img)
        if tops[i] is not None:
            images.append(zoomed_img)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    num_current_image = 0
    output_images = []

    for image in images:
        num_current_image += 1

        image = image[None, :].cuda()
        model_kwargs = {}

        with th.no_grad():
            feat = classification_model(image).detach()
            model_kwargs["condition"] = feat

            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        batch = ((image[0:1] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()
        output_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in batch])

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        samples = sample.contiguous()

        output_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
        logger.log(f"created {args.batch_size*num_current_image} samples")

    arr = np.concatenate(output_images, axis=0)
    save_image(th.FloatTensor(arr).permute(0, 3, 1, 2), args.out_dir + '/' + args.name + '.jpeg', normalize=True,
               scale_each=True, nrow=args.batch_size+1)
    print('Image saved at: /home/deep-learning-advanced-course/results/sample_images/'+args.name+'.jpeg')
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path="",
        out_dir="../../../results/sample_images",
        name='{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()),
        use_dino=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()