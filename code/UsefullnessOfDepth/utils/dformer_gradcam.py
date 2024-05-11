import argparse
import importlib
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import sys
sys.path.append('../UsefullnessOfDepth')

from model_DFormer.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import get_train_loader,get_val_loader
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model, cfg=None): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        self.cfg = cfg
        
    def forward(self, x):
        if self.cfg is None:
            if x.shape[1] == 2:
                return self.model(x[:, 0], x[:, 1])
            elif x.shape[1] >= 4:
                rgb = x[:, :3, :, :]
                depth = x[:, 3, :, :]
                return self.model(rgb, depth)
            else:
                return self.model(x)
        else:
            x_channels = self.cfg.x_channels
            rgb = x[:, :x_channels, :, :]
            depth = x[:, x_channels:, :, :]
            return self.model(rgb, depth)

    
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

class DFormerAnalyzer:
    def __init__(self, model, cfg=None):
        self.model = SegmentationModelOutputWrapper(model, cfg=cfg)
        self.downsample_layers = [
            model.backbone.downsample_layers[0],
            model.backbone.downsample_layers[1],
            model.backbone.downsample_layers[2],
            model.backbone.downsample_layers[3],
        ]
        self.downsample_layers_e = [
            model.backbone.downsample_layers_e[0],
            model.backbone.downsample_layers_e[1],
            model.backbone.downsample_layers_e[2],
            model.backbone.downsample_layers_e[3],
        ]

        self.decode_layers = [
            model.decode_head.conv_seg,
            model.decode_head.squeeze.conv,
            model.decode_head.hamburger,
            model.decode_head.hamburger.ham_in.conv,
            model.decode_head.hamburger.ham_out.conv,
            model.decode_head.align.conv,
        ]

    def create_mask(self, category, input_tensor):
        prediction_tensor = self.model(input_tensor)
        prediction_np = prediction_tensor[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        return np.float32(prediction_np == category)
    
    def create_target(self, category, mask):
        return SemanticSegmentationTarget(category, mask)
    
    def select_category(self, categories_of_interest, prediction_np):
        assert len(categories_of_interest) > 0
        assert isinstance(prediction_np, np.ndarray)

        if len(categories_of_interest) > 1:
            background_class = 0
            categories_of_interest = [category for category in categories_of_interest if category != background_class]

        unique_classes = np.unique(prediction_np)
        for category in categories_of_interest:
            if category in unique_classes:
                return category
            
        return np.random.choice(unique_classes)
        
    def analyze(self, input_tensor, category, target_layers=None):
        mask = self.create_mask(category, input_tensor)
        target = self.create_target(category, mask)

        image = input_tensor[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        if target_layers is None:
            target_layers = self.downsample_layers + self.downsample_layers_e + self.decode_layers
        with GradCAM(model=self.model,
                    target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor.clone(),
                                targets=[target])[0, :]
            cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
            return cam_image
        
def set_config_if_dataset_specified(config, dataset_location):
    config.dataset_path = dataset_location
    config.rgb_root_folder = os.path.join(config.dataset_path, 'RGB')
    config.gt_root_folder = os.path.join(config.dataset_path, 'labels')
    config.x_root_folder = os.path.join(config.dataset_path, 'Depth')
    config.train_source = os.path.join(config.dataset_path, "train.txt")
    config.eval_source = os.path.join(config.dataset_path, "test.txt")
    return config

def normalize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image

def analyzeDataset(model_weights_dir, dataset_dir, config, target_layers=None, target_categories=[]):
    config = set_config_if_dataset_specified(config, dataset_dir)
    model = segmodel(cfg=config, criterion=nn.CrossEntropyLoss(reduction='mean'), norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(model_weights_dir)["model"])
    model = model.eval()
    model = model.to(device)
    analyzer = DFormerAnalyzer(model)
    dataloader, _ = get_val_loader(None, RGBXDataset, config)

    dataset_name = os.path.basename(dataset_dir.rstrip('\\'))
    dataset_name = dataset_name.replace("SynthDet_", "")

    grad_results_dir = os.path.join(os.path.dirname(model_weights_dir), f"gradcam_results_for_{dataset_name}")
    if not os.path.exists(grad_results_dir):
        os.makedirs(grad_results_dir)

    if target_layers is None:
        target_layers = analyzer.downsample_layers + analyzer.downsample_layers_e + analyzer.decode_layers

    for i, minibatch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        image_tensor = minibatch["data"]
        depth_tensor = minibatch["modal_x"]
        target_tensor = minibatch["label"]
        input_tensor = torch.cat([image_tensor, depth_tensor], dim=1).to(device)
        prediction = model(input_tensor[:, :3], input_tensor[:, 3:])
        prediction_np = prediction[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        unique_classes = np.unique(prediction_np)
        if len(target_categories) > 0:
            unique_classes = [category for category in unique_classes if category in target_categories]
        for category in unique_classes:
            grad_results_category_dir = os.path.join(grad_results_dir, f"category_{category}")
            if not os.path.exists(grad_results_category_dir):
                os.makedirs(grad_results_category_dir)

            prediction_mask = np.where(prediction_np == category, 1, 0)
            target_mask = np.where(target_tensor[0].numpy() == category, 1, 0)

            target_image = np.stack([target_mask]*3, axis=-1) * np.array([0, 255, 0])
            prediction_image = np.stack([prediction_mask]*3, axis=-1) * np.array([255, 0, 0])

            cam_image = analyzer.analyze(input_tensor, category, target_layers=target_layers)
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            ax[0, 0].imshow(normalize_image(image_tensor[0].permute(1, 2, 0).numpy()))
            ax[0, 0].set_title("RGB Image")
            ax[0, 0].axis('off')

            ax[0, 1].imshow(depth_tensor[0].permute(1, 2, 0).numpy())
            ax[0, 1].set_title("Depth Image")
            ax[0, 1].axis('off')

            ax[1, 0].imshow(np.clip(prediction_image + target_image, 0, 255).astype(np.uint8))
            ax[1, 0].set_title("Prediction vs Target")
            # Create a Patch object for each label in the legend
            prediction_patch = mpatches.Patch(color='red', label='Prediction')
            target_patch = mpatches.Patch(color='green', label='Target')
            correct_patch = mpatches.Patch(color='yellow', label='Correct')

            # Add the patches to the legend
            ax[1, 0].legend(handles=[prediction_patch, target_patch, correct_patch])
            ax[1, 0].axis('off')

            ax[1, 1].imshow(cam_image)
            ax[1, 1].set_title("GradCAM")
            ax[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(grad_results_category_dir, f"gradcam_{i}.png"))
    


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-cfg', '--config', type=str)
    argparser.add_argument('-mp', '--model_path', type=str)
    argparser.add_argument('-d', '--dataset_dir', type=str)
    argparser.add_argument('-a', '--analyze_dataset', action='store_true')
    argparser.add_argument('-t', '--target_categories', type=int, nargs='+', default=[])

    args = argparser.parse_args()
    module_name = args.config
    if ".py" in module_name:
        module_name = module_name.replace(".py", "")
        module_name = module_name.replace("\\", ".")
        while module_name.startswith("."):
            module_name = module_name[1:]

    args.config = module_name

    config_module = importlib.import_module(args.config)
    config = config_module.config

    if args.analyze_dataset:
        analyzeDataset(args.model_path, args.dataset_dir, config, target_categories=args.target_categories)
        exit()

     
    val_loader, val_sampler = get_val_loader(None, RGBXDataset,config,1)

    dataloader = iter(val_loader)
    data = next(dataloader)
    image_tensor = data['data']
    depth_tensor = data['modal_x']

    image = image_tensor[0].permute(1, 2, 0).numpy()
    depth = depth_tensor[0].permute(1, 2, 0).numpy()

    image = (image - image.min()) / (image.max() - image.min())
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    image = image.astype(np.float32)
    depth = depth.astype(np.float32)

    # input_tensor = torch.stack([image_tensor, depth_tensor], dim=1)
    input_tensor = torch.cat([image_tensor, depth_tensor], dim=1)

    # Setup model
    criterion = nn.CrossEntropyLoss(reduction='mean')
    config.pretrained_model = args.model_path
    model=segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(args.model_path)["model"])
    model = model.eval()

    # with open('layers.txt', 'w') as file:
    #     file.write("All layers and their names:\n")
    #     for name, module in model.named_modules():
    #         file.write(f"{name}\n")
    
    # exit()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
    
    # model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor[:, 0], input_tensor[:, 1])

    # normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    normalized_masks = output.cpu()

    prediction = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()

    sem_classes = config.class_names

    class_names_in_prediction = [sem_classes[i] for i in np.unique(prediction)]
    print("class names in prediction: ", class_names_in_prediction)

    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    category = sem_class_to_idx["SoftStar"]
    print("Chosen category: ", category)
    
    # car_category = 28
    car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    car_mask_uint8 = 255 * np.uint8(car_mask == category)
    car_mask_float = np.float32(car_mask == category)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(image)
    ax[1].imshow(car_mask_uint8, cmap='viridis')
    ax[2].imshow(prediction, cmap='viridis')
    plt.show()
            
    analyzer = DFormerAnalyzer(model)
    cam_image_downsample = analyzer.analyze(input_tensor, category, target_layers=analyzer.downsample_layers)
    cam_image_downsample_e = analyzer.analyze(input_tensor, category, target_layers=analyzer.downsample_layers_e)
    cam_image_decode = analyzer.analyze(input_tensor, category, target_layers=analyzer.decode_layers)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(cam_image_downsample)
    ax[0].set_title("Downsample layers")
    ax[1].imshow(cam_image_downsample_e)
    ax[1].set_title("Downsample layers_e")
    ax[2].imshow(cam_image_decode)
    ax[2].set_title("Decode layers")
    plt.show()
