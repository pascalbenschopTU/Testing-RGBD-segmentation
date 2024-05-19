from functools import partial
import os
import sys

sys.path.append('../UsefullnessOfDepth')

from torch.utils.data import DataLoader
import argparse
import importlib
import torch
import torch.nn as nn
from utils.dataloader.dataloader import get_train_loader,get_val_loader, ValPre
# Model
from model_DFormer.builder import EncoderDecoder as segmodel
from models_CMX.builder import EncoderDecoder as cmxmodel
from model_pytorch_deeplab_xception.deeplab import DeepLab
from models_segformer import SegFormer

# Dataset
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.metrics_new import Metrics
from tensorboardX import SummaryWriter
import random
import numpy as np
from functools import partial

# Ray imports for hyperparameter tuning
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray import train
from ray.air.integrations.wandb import WandbLoggerCallback


# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# https://docs.ray.io/en/latest/tune/examples/hpo-frameworks.html
# https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-suggest-optuna

    
def evaluate(model, dataloader, config, device, criterion=nn.CrossEntropyLoss(reduction='mean')):
    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, 255, device)

    val_loss = 0.0
    val_steps = 0

    for minibatch in dataloader:
        images = minibatch["data"][0]
        labels = minibatch["label"][0]
        modal_xs = minibatch["modal_x"][0]
        # print(images.shape,labels.shape)
        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        predictions = model(images[0], images[1])
        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)
        # print(preds.shape,labels.shape)
        metrics.update(predictions.softmax(dim=1), labels)
        loss = criterion(predictions, labels.long())
        val_loss += loss.item()
        val_steps += 1

    return metrics, val_loss / val_steps

def get_dataset(config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = ValPre(config.norm_mean, config.norm_std,config.x_is_single_channel,config)

    num_imgs = (config.num_train_imgs // config.batch_size + 1) * config.batch_size
    train_dataset = RGBXDataset(data_setting, "train", train_preprocess, num_imgs)

    return train_dataset

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = (
        True  # train speed is slower after enabling this opts.
    )

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)

def train_dformer(config, original_config, train_dataset, num_epochs=5, max_dataset_size=1000):
    hyperparameters = config.copy()
    
    config = original_config
    # Set hyperparemeters
    config.lr = hyperparameters["lr"]
    config.lr_power = hyperparameters["lr_power"]
    config.momentum = hyperparameters["momentum"]
    config.weight_decay = hyperparameters["weight_decay"]
    config.batch_size = hyperparameters["batch_size"]
    config.nepochs = num_epochs
    config.warm_up_epoch = 1

    set_seed(config.seed)

    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, 
        [max_dataset_size, len(train_dataset) - max_dataset_size],
    )

    dataset_length = len(train_dataset)
    train_length = int(0.7 * dataset_length)
    validate_length = dataset_length - train_length

    train_subset, validate_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_length, validate_length],
    )
    train_size = len(train_subset)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(validate_subset, batch_size=config.batch_size, shuffle=False)
    print('train size: ',len(train_loader), 'val size: ',len(val_loader), "batch size: ", config.batch_size)

    # Dont ignore the background class
    criterion = nn.CrossEntropyLoss(reduction='mean')
    BatchNorm2d = nn.BatchNorm2d
    
    if "DFormer" in config.backbone:
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "mit_b2":
        model = cmxmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "xception":
        model = DeepLab(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "segformer":
        model = SegFormer(cfg=config, criterion=criterion)
    
    base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if hyperparameters["optimizer"] == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif hyperparameters["optimizer"] == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    config.niters_per_epoch = train_size // config.batch_size + 1
    if train_size % config.batch_size == 0:
        config.niters_per_epoch = train_size // config.batch_size
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ',device)
    model.to(device)

    optimizer.zero_grad()
    best_miou=0.0
    
    for epoch in range(1, config.nepochs+1):
        model.train()
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = range(config.niters_per_epoch)
        dataloader = iter(train_loader)

        sum_loss = 0
        i=0
        for idx in pbar:
            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']
            if len(imgs) < config.batch_size:
                continue

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            # loss = model(imgs, modal_xs, gts)
            output = model(imgs, modal_xs)
            loss = criterion(output, gts.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += loss

        with torch.no_grad():
            model.eval()
            device = torch.device('cuda')
            metric, loss = evaluate(model, val_loader, config, device, criterion=criterion)
            ious, miou = metric.compute_iou()
            acc, macc = metric.compute_pixel_acc()
            f1, mf1 = metric.compute_f1()

            if miou > best_miou:
                best_miou = miou
            
        print('macc: ', macc, 'mf1: ', mf1, 'miou: ',miou,'best: ',best_miou)

        train.report({"miou": miou, "loss": loss})


def shorten_trial_dirname_creator(trial, experiment_name="empty"):
    # Extract relevant trial information and create a shortened directory name
    short_trial_name = f"{experiment_name}_{trial.trial_id}"
    return short_trial_name

def update_config_paths(original_config):
    # Make all paths in original_config absolute
    original_config.dataset_path = os.path.abspath(original_config.dataset_path)
    original_config.rgb_root_folder = os.path.abspath(original_config.rgb_root_folder)
    original_config.gt_root_folder = os.path.abspath(original_config.gt_root_folder)
    original_config.x_root_folder = os.path.abspath(original_config.x_root_folder)
    original_config.log_dir = os.path.abspath(original_config.log_dir)
    original_config.tb_dir = os.path.abspath(original_config.tb_dir)
    original_config.checkpoint_dir = os.path.abspath(original_config.checkpoint_dir)
    original_config.train_source = os.path.abspath(original_config.train_source)
    original_config.eval_source = os.path.abspath(original_config.eval_source)
    
    return original_config

def tune_hyperparameters(original_config, num_samples=20, max_num_epochs=5, cpus_per_trial=16, gpus_per_trial=1):
    original_config = update_config_paths(original_config)

    experiment_name = f"{original_config.dataset_name}_{original_config.backbone}"

    param_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([4, 8, 16]),
        "lr_power": tune.uniform(0.85, 1.0),
        "momentum": tune.uniform(0.85, 0.99),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "optimizer": tune.choice(["AdamW", "SGDM"]),
    }

    large_models = ["mit_b2", "xception"]
    if original_config.backbone in large_models:
        param_space["batch_size"] = tune.choice([4, 8])
    extra_large_models = ["DFormer-Large"]
    if original_config.backbone in extra_large_models:
        param_space["batch_size"] = tune.choice([4])
    
    algorithm = OptunaSearch(
        metric="miou",
        mode="max",
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="miou",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    train_dataset = get_dataset(original_config)
    import torch
    import torch._dynamo
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_dformer, 
                original_config=original_config,
                train_dataset=train_dataset,
                num_epochs=max_num_epochs,                
            ),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=algorithm,
            trial_dirname_creator=partial(shorten_trial_dirname_creator, experiment_name=experiment_name),
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            # callbacks=[WandbLoggerCallback(
            #     project="DFormer_hyperparameter_tuning",
            #     group=experiment_name,
            #     api_key=os.environ.get("WANDB_API_KEY"),
            # )],
            stop={"training_iteration": max_num_epochs},
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    best_config = results.get_best_result(
        metric="miou",
        mode="max",    
    ).config
    print("Best hyperparameters found were: ", best_config)

    ray.shutdown()

    original_config.lr = best_config["lr"]
    original_config.lr_power = best_config["lr_power"]
    original_config.momentum = best_config["momentum"]
    original_config.weight_decay = best_config["weight_decay"]
    original_config.batch_size = best_config["batch_size"]
    original_config.optimizer = best_config["optimizer"]

    return original_config, best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="DFormer.local_configs.SynthDet.SynthDet_black_back_default_2_Dformer_Tiny",
        help="The config to use for training the model",
    )
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    original_config = config_module.config

    final_config = tune_hyperparameters(original_config, num_samples=20, max_num_epochs=5, cpus_per_trial=16, gpus_per_trial=1, experiment_name="empty")