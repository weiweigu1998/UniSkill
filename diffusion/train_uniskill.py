#!/usr/bin/env python

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoImageProcessor, CLIPTextModel, AutoModelForDepthEstimation

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from dataset import *
from pipeline_dynamics import DiffusionPix2PixPipelineDynamics

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dynamics.idm import IDM


if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="UniSkill."
    )
    group = parser.add_mutually_exclusive_group(required=False)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help="Variant of the model files of the pretrained model identifier"
        " from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. "
            "Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outptus",
        help="The output directory where the model"
        "predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images"
            "in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--idm_resolution",
        type=int,
        default=224,
        help=(
            "The resolution for depth"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  "
        "If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates."
            "Checkpoints can be used for resuming training via "
            "`--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, "
            "the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of"
            "the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint"
            "for step by step instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. "
            "Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to '
            "automatically select the last available checkpoint."
        ),
    )
    group.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. "
            "Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to '
            "automatically select the last available checkpoint."
            "This argument cannot be used with `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate"
        " before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing "
        "to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, "
        "gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", '
            '"cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. "
            "0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
            "Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. "
            "Can be used to speed up training. "
            "For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            "The integration to report the results and logs to."
            'Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. '
            'Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="uniskill",
        help=("Name of wandb run"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). "
            "Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  "
            "Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. "
            "Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. "
            "Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, "
            "truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, "
        "`--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_uniskill",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="droid",
        help="The dataset to use for training.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset directory. Overrides the dataset class default.",
    )
    parser.add_argument(
        "--train_scheduler",
        type=str,
        default="ddpm",
        help=("Type of scheduler to train.", "Options: ddpm, euler_discrete, ddim"),
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="full",
        help=("Spacing of timesteps", "Options: full, turbo_timesteps"),
    )
    parser.add_argument(
        "--do_classifier_free_guidance",
        action="store_true",
        help=("Use classifier-free guidance."),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def log_validation(
    valid_dataset,
    vae,
    text_encoder,
    tokenizer,
    depth_estimator,
    generator,
    args,
    accelerator,
    weight_dtype,
    unet,
    idm,
):
    logger.info("Running validation... ")
    
    unet = accelerator.unwrap_model(unet)
    idm = accelerator.unwrap_model(idm)

    pipeline = DiffusionPix2PixPipelineDynamics.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    # Select scheduler
    if args.train_scheduler == "ddpm":
        scheduler_cls = DDPMScheduler
    elif args.train_scheduler == "euler_discrete":
        scheduler_cls = EulerDiscreteScheduler
    elif args.train_scheduler == "ddim":
        scheduler_cls = DDIMScheduler
    else:
        raise ValueError(f"Scheduler {args.train_scheduler} not supported")
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    

    for _ in range(args.num_validation_images):
        idx = random.randint(0, len(valid_dataset) - 1)
        sampled_example = valid_dataset[idx]
        
        curr_depth_feat = sampled_example["curr_depth_features"]
        next_depth_feat = sampled_example["next_depth_features"]

        curr_img = sampled_example["curr_images"]
        next_img = sampled_example["next_images"]
        
        idm_curr_img = sampled_example["idm_curr_images"]
        idm_next_img = sampled_example["idm_next_images"]
        
        idm_curr_img = idm_curr_img.to(accelerator.device, dtype=weight_dtype).unsqueeze(0)
        idm_next_img = idm_next_img.to(accelerator.device, dtype=weight_dtype).unsqueeze(0)
        idm_visual_pair = torch.stack([idm_curr_img, idm_next_img], dim=1)
        
        depth_feat = torch.tensor(np.stack([curr_depth_feat, next_depth_feat], axis=0), device=accelerator.device)
        depth_feat = depth_estimator(depth_feat).predicted_depth
        depth_min, depth_max = depth_feat.flatten(1).min(-1).values, depth_feat.flatten(1).max(-1).values
        depth_feat = (depth_feat - depth_min[..., None, None]) / (depth_max - depth_min)[..., None, None]
        curr_depth_features, next_depth_features = torch.chunk(depth_feat, 2, dim=0)

        depth_pair = torch.stack([curr_depth_features, next_depth_features], dim=1)
        depth_pair = F.interpolate(depth_pair, size=(args.idm_resolution, args.idm_resolution), mode="bilinear", align_corners=False)

        images, noisy_images, errors = [], [], []
        image_logs = []
        latent_action = idm(depth_pair, idm_visual_pair)
        prompt_embeds = latent_action

        num_samples = 1
        for _ in range(num_samples):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt_embeds=prompt_embeds,
                    image=curr_img,
                    num_inference_steps=30,
                    generator=generator,
                    guidance_scale=7.5,
                    image_guidance_scale=2.5,
                ).images[0]

                noisy_image = pipeline(
                    prompt_embeds=torch.zeros_like(prompt_embeds),
                    image=curr_img,
                    num_inference_steps=30,
                    generator=generator,
                    guidance_scale=0.0,
                ).images[0]

            images.append(image)
            noisy_images.append(noisy_image)

            difference_image = np.array(image) - np.array(next_img)
            mse = np.mean(np.square(difference_image))
            if mse > 0:
                norm_mse = difference_image / np.sqrt(mse)
            else:
                norm_mse = difference_image
            norm_mse *= 255

            errors.append(norm_mse)

        image_logs.append(
            {
                "validation_image": curr_img,
                "gt_image": next_img,
                "images": images,
                "noisy_images": noisy_images,
                "errors": errors,
                "mse": mse,
            }
        )
    
    for tracker in accelerator.trackers:
        formatted_images = []
        formatted_errors = []

        mse_errors = []
        for log in image_logs:
            images = log["images"]
            noisy_images = log["noisy_images"]
            errors = log["errors"]
            validation_image = log["validation_image"]
            gt_image = log["gt_image"]
            mse = log["mse"]

            mse_errors.append(mse)

            formatted_images.append(
                wandb.Image(validation_image, caption="Conditioning Image")
            )
            formatted_images.append(
                wandb.Image(gt_image, caption="Ground Truth Target")
            )

            for image in images:
                image = wandb.Image(image, caption="Generated Image")
                formatted_images.append(image)

            for noisy_image in noisy_images:
                noisy_image = wandb.Image(noisy_image, caption="Noisy Generated Image")
                formatted_images.append(noisy_image)

            for error in errors:
                error = wandb.Image(error, caption="Normalized MSE")
                formatted_errors.append(error)

        tracker.log({"validation": formatted_images})
        tracker.log({"val_mse": np.mean(mse_errors)})

        return image_logs

def make_dataset(dataset_name, args, depth_processor, train=True):
    dataset_classes = {
        "xskill": XSkillDataset,
        "sthsthv2": SthSthv2Dataset,
        "bridge": BridgeDataset,
        "droid": DroidDataset,
        "libero": LIBERODataset,
        "bridge": BridgeDataset,
        "action_bench": ActionBenchDataset,
        "combined": CombinedDataset,
    }

    dataset_class = dataset_classes.get(dataset_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    common_kwargs = dict(
        train=train,
        resolution=args.resolution,
        depth_processor=depth_processor,
    )
    if getattr(args, "data_path", None) is not None:
        common_kwargs["data_path"] = args.data_path

    if dataset_name == "combined":
        common_kwargs.pop("data_path", None)
        return dataset_class(
            **common_kwargs,
            datasets=["droid", "libero", "bridge", "sthsthv2", "h2o"],
        )
    elif dataset_name == "xskill":
        return dataset_class(
            **common_kwargs,
            unseen_type="human",
        )
    else:
        return dataset_class(**common_kwargs)

def collate_fn(examples):
    curr_images = torch.stack([example["curr_images"] for example in examples])
    curr_images = curr_images.to(memory_format=torch.contiguous_format).float()

    next_images = torch.stack([example["next_images"] for example in examples])
    next_images = next_images.to(memory_format=torch.contiguous_format).float()
    
    idm_curr_images = torch.stack([example["idm_curr_images"] for example in examples])
    idm_curr_images = idm_curr_images.to(memory_format=torch.contiguous_format).float()
    
    idm_next_images = torch.stack([example["idm_next_images"] for example in examples])
    idm_next_images = idm_next_images.to(memory_format=torch.contiguous_format).float()
    
    curr_depth_features = torch.stack([torch.tensor(example["curr_depth_features"]) for example in examples])
    curr_depth_features = curr_depth_features.to(memory_format=torch.contiguous_format).float()
    
    next_depth_features = torch.stack([torch.tensor(example["next_depth_features"]) for example in examples])
    next_depth_features = next_depth_features.to(memory_format=torch.contiguous_format).float()

    return {
        "curr_images": curr_images,
        "next_images": next_images,
        "idm_curr_images": idm_curr_images,
        "idm_next_images": idm_next_images,
        "curr_depth_features": curr_depth_features,
        "next_depth_features": next_depth_features,
    }

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load scheduler and models
    if args.train_scheduler == "ddpm":
        schduler_cls = DDPMScheduler
    elif args.train_scheduler == "euler_discrete":
        schduler_cls = EulerDiscreteScheduler
    elif args.train_scheduler == "ddim":
        schduler_cls = DDIMScheduler
    else:
        raise ValueError(f"Scheduler {args.train_scheduler} not supported.")
    noise_scheduler = schduler_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    depth_estimator = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    
    idm = IDM(
        num_layers=8,
        num_heads=4,
        hidden_dim=256,
        skill_dim=64,
        out_dim=unet.config.cross_attention_dim,
        idm_resolution=args.idm_resolution,
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that
        # `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if False:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    name = model.__class__.__name__.lower()
                    if name == "idm":
                        torch.save(model.state_dict(), f"{output_dir}/idm.pth")
                    elif name == "unet2dconditionmodel":
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif name == "actionadapter":
                        torch.save(model.state_dict(), f"{output_dir}/action_adapter.pth")

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()
                name = model.__class__.__name__.lower()
                if name == "idm":
                    state_dict = torch.load(f"{input_dir}/idm.pth", map_location='cpu')
                    model.load_state_dict(state_dict)
                elif name == "unet2dconditionmodel":
                    load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    depth_estimator.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in \
            full float32 precision when starting training - even if"
        " doing mixed precision training, \
            copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"unet loaded as datatype \
            {accelerator.unwrap_model(unet).dtype}. \
            {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: \
                    `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(unet.parameters()) + list(idm.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Train dataloader
    train_dataset = make_dataset(args.dataset_name, args, depth_processor, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    valid_dataset = make_dataset(args.dataset_name, args, depth_processor, train=False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, idm, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, idm, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder
    # and vae weights to half-precision
    # as these models are only used for inference,
    # keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    depth_estimator.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps
    # as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        
        accelerator.init_trackers(
            "train_uniskill",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f'{args.dataset_name}/{args.report_name}'}},
        )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, \
            distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.finetune_from_checkpoint:
        assert os.path.exists(args.finetune_from_checkpoint), f"Checkpoint '{args.finetune_from_checkpoint}' does not exist. Starting a new training run."
        accelerator.print(f"Finetune from checkpoint {args.finetune_from_checkpoint}")
        accelerator.load_state(args.finetune_from_checkpoint)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' \
                    does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if args.do_classifier_free_guidance:
        uncond_prob = 0.05

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([unet, idm]):
                # Data augmentation
                latents = vae.encode(
                    batch["next_images"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                curr_latents = vae.encode(
                    batch["curr_images"].to(dtype=weight_dtype)
                ).latent_dist.mode()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents
                # according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                depth_features = torch.cat([
                    batch["curr_depth_features"].to(dtype=weight_dtype),
                    batch["next_depth_features"].to(dtype=weight_dtype),
                ])
                with torch.no_grad():
                    depth_outputs = depth_estimator(depth_features).predicted_depth
                    depth_min, depth_max = depth_outputs.flatten(1).min(dim=1)[0], depth_outputs.flatten(1).max(dim=1)[0]
                    depth_outputs = (depth_outputs - depth_min[..., None, None]) / (depth_max - depth_min)[..., None, None]
                curr_depth_features, next_depth_features = torch.chunk(depth_outputs, 2, dim=0)

                depth_pair = torch.stack([curr_depth_features, next_depth_features], dim=1)
                depth_pair = F.interpolate(depth_pair, size=(args.idm_resolution, args.idm_resolution), mode="bilinear", align_corners=False)
                
                visual_pair = torch.stack([
                    batch["idm_curr_images"].to(dtype=weight_dtype),
                    batch["idm_next_images"].to(dtype=weight_dtype)
                ], dim=1)
                
                latent_action = idm(depth_pair, visual_pair)
                
                prompt_mask = None
                if args.do_classifier_free_guidance:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator
                    )
                    prompt_mask = random_p >= 2 * uncond_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    latent_action = latent_action * prompt_mask

                    image_mask = 1 - (
                        ((random_p >= uncond_prob) * (random_p < 3 * uncond_prob)).to(curr_latents.dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    curr_latents = curr_latents * image_mask
                    
                prompt_embeds = latent_action
                concatenated_latents = torch.cat([noisy_latents, curr_latents], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    concatenated_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type \
                            {noise_scheduler.config.prediction_type}"
                    )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters()) + list(idm.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed
            # an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save
                        # would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have
                            # at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, \
                                        removing {len(removing_checkpoints)} \
                                            checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: \
                                        {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            valid_dataset,
                            vae,
                            text_encoder,
                            tokenizer,
                            depth_estimator,
                            generator,
                            args,
                            accelerator,
                            weight_dtype,
                            unet,
                            idm,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        idm = accelerator.unwrap_model(idm)
        torch.save(idm.state_dict(), os.path.join(args.output_dir, "idm.pth"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
