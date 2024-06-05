from functools import partial
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import argparse
import importlib
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import random
import numpy as np
from functools import partial

sys.path.append('../UsefullnessOfDepth')

# Dataset
from utils.dataloader.dataloader import get_train_loader,get_val_loader, ValPre
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.metrics_new import Metrics

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

def ray_callback(miou, loss, epoch):
    train.report({"miou": miou, "loss": loss, "epoch": epoch})

def train_dformer(hyperparameters, config, train_dataset, num_epochs=5, train_callback=None):
    # Set hyperparemeters
    config.lr = hyperparameters["lr"]
    config.lr_power = hyperparameters["lr_power"]
    config.momentum = hyperparameters["momentum"]
    config.weight_decay = hyperparameters["weight_decay"]
    config.batch_size = hyperparameters["batch_size"]
    config.optimizer = hyperparameters["optimizer"]
    config.nepochs = num_epochs
    config.warm_up_epoch = 1

    # set_seed(config.seed)

    dataset_length = len(train_dataset)
    train_length = int(0.8 * dataset_length)
    validate_length = dataset_length - train_length

    train_subset, validate_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_length, validate_length],
    )
    train_size = len(train_subset)

    config.num_train_imgs = train_size
    config.checkpoint_step = 1
    config.checkpoint_start_epoch = 0

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(validate_subset, batch_size=config.batch_size, shuffle=False)
    print('train size: ',len(train_loader), 'val size: ',len(val_loader), "batch size: ", config.batch_size)

    kwargs = {
        "is_tuning": True,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "ray_callback": ray_callback,
    }

    if train_callback is None:
        print("No train callback provided, exiting")
        return
    train_callback(config, **kwargs)


def shorten_trial_dirname_creator(trial, experiment_name="empty"):
    # Extract relevant trial information and create a shortened directory name
    short_trial_name = f"{experiment_name}_{trial.trial_id}"
    return short_trial_name

def update_config_paths(config):
    # Make all paths in config absolute
    config.dataset_path = os.path.abspath(config.dataset_path)
    config.rgb_root_folder = os.path.abspath(config.rgb_root_folder)
    config.gt_root_folder = os.path.abspath(config.gt_root_folder)
    config.x_root_folder = os.path.abspath(config.x_root_folder)
    config.log_dir = os.path.abspath(config.log_dir)
    config.tb_dir = os.path.abspath(config.tb_dir)
    config.checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    config.train_source = os.path.abspath(config.train_source)
    config.eval_source = os.path.abspath(config.eval_source)
    if config.pretrained_model is not None:
        config.pretrained_model = os.path.abspath(config.pretrained_model)
    
    return config

def tune_hyperparameters(config, num_samples=20, max_num_epochs=5, cpus_per_trial=4, gpus_per_trial=1, train_callback=None):
    config = update_config_paths(config)

    experiment_name = f"{config.dataset_name}_{config.backbone}"

    param_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([8, 16]),
        "lr_power": tune.uniform(0.8, 1.0),
        "momentum": tune.uniform(0.9, 0.99),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "optimizer": tune.choice(["AdamW", "SGDM"]),
    }

    large_models = ["mit_b2", "xception", "mit_b3"]
    if config.backbone in large_models:
        param_space["batch_size"] = tune.choice([4, 8])
    extra_large_models = ["DFormer-Large"]
    if config.backbone in extra_large_models:
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

    train_dataset = get_dataset(config)
    import torch
    import torch._dynamo
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    ray.init()

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_dformer, 
                config=config,
                train_dataset=train_dataset,
                num_epochs=max_num_epochs,
                train_callback=train_callback,                
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

    return best_config
