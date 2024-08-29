from functools import partial
import os
import sys
from torch.utils.data import DataLoader
import torch
import numpy as np
from functools import partial

sys.path.append('../UsefullnessOfDepth')

# Dataset
from utils.dataloader.dataloader import ValPre
from utils.dataloader.RGBXDataset import RGBXDataset

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

    num_imgs = int((config.num_train_imgs // config.batch_size + 1) * config.batch_size)
    train_dataset = RGBXDataset(data_setting, "train", train_preprocess, num_imgs)

    return train_dataset

def ray_callback(miou, loss, epoch):
    train.report({"miou": miou, "loss": loss, "epoch": epoch})

def train_dformer(hyperparameters, config, train_dataset, num_epochs=5, train_callback=None):
    # Set hyperparemeters
    config.lr = hyperparameters["lr"]
    config.lr_power = hyperparameters["lr_power"]
    config.momentum = hyperparameters["momentum"]
    config.weight_decay = hyperparameters["weight_decay"]
    config.batch_size = hyperparameters["batch_size"]
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

    config.num_train_imgs = (train_size // config.batch_size) * config.batch_size
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
    }

    model = config.get("model", None)
    large_models = ["mit_b2", "xception", "mit_b3", "DFormer-Base", "TokenFusion", "Gemini", "CMX", "HIDANet"]
    if config.backbone in large_models or model in large_models:
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
