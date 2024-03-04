import argparse
import importlib
import os
import pathlib
import sys
import time
import cv2

from tqdm import tqdm
sys.path.append("../../DFormer/utils/dataloader/")
from smart_model import SmartPeripheralRGBDModel, SmallUNet, SmartDepthModel
from RGBXDataset import RGBXDataset
from smart_dataloader import get_train_loader, get_val_loader
from utils import group_weight, WarmUpPolyLR

import torch
import torch.nn as nn
import torch.nn.init as init
from tensorboardX import SummaryWriter
from eval import Metrics

# N_CLASSES = 81
N_CLASSES = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train(config, args):
    """
    :param dataset_location: path to the dataset
    :param epochs: number of epochs
    :param batch_size: batch size
    :param learning_rate: learning rate
    :param device: device to use for training
    """

    config_module = importlib.import_module(args.config)
    config = config_module.config

    config.log_dir += '/c'+str(args.channels)
    config.log_dir += '_'+time.strftime('%Y%m%d-%H%M%S', time.localtime()).replace(' ','_')
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    tb_dir = config.tb_dir+'/_'+time.strftime('%Y%m%d-%H%M%S', time.localtime()).replace(' ','_')
    tb = SummaryWriter(tb_dir)
    
    train_loader = get_train_loader(RGBXDataset, config)
    val_loader = get_val_loader(RGBXDataset, config)
    # Create the model
    # model = SmartPeripheralRGBDModel(args.channels, N_CLASSES, criterion=torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background))
    model = SmallUNet(args.channels, N_CLASSES, criterion=torch.nn.CrossEntropyLoss(reduction='mean'), config=config, writer=tb)
    # model = SmartDepthModel(args.channels, N_CLASSES, criterion=torch.nn.CrossEntropyLoss(reduction='mean'), config=config, writer=tb)
    
    # Initialize the weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    # print amount of parameters
    print('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    base_lr = config.lr

    param_list = []
    param_list = group_weight(param_list, model, nn.BatchNorm2d, base_lr)

    optimizer = torch.optim.Adam(param_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    total_iteration = config.niters_per_epoch * config.nepochs
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.warm_up_epoch * config.niters_per_epoch)

    model.to(device)
    optimizer.zero_grad()
    best_miou = 0.0

    for epoch in range(config.start_epoch, config.nepochs+1):
        model.train()
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(int(len(train_loader))), file=sys.stdout,
                bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        i=0
        for idx in pbar:
            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.to(device)
            gts = gts.to(device)
            modal_xs = modal_xs.to(device)

            loss = model(imgs, modal_xs, gts)

            if epoch % 10 == 0 and idx == 0 or epoch == 1 and idx == 0:
                with torch.no_grad():
                    model(imgs, modal_xs, gts, plot=True, epoch=epoch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += loss
            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            
            tb.add_scalar('train/loss', loss, current_idx)
            
            pbar.set_description(print_str, refresh=False)
            pbar.update(1)
        
        if (epoch%config.checkpoint_step==0 and epoch>int(config.checkpoint_start_epoch)):
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                model.eval()

                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(int(len(val_loader))), file=sys.stdout,
                            bar_format=bar_format)
                dataloader = iter(val_loader)

                n_classes = config.num_classes
                metric = Metrics(n_classes, config.background, device)

                for idx in pbar:
                    minibatch = dataloader.next()
                    images_eval = minibatch['data']
                    labels_eval = minibatch['label']
                    modal_xs_eval = minibatch['modal_x']

                    images_eval = images_eval.to(device)
                    labels_eval = labels_eval.to(device)
                    modal_xs_eval = modal_xs_eval.to(device)

                    logits = model(images_eval, modal_xs_eval)

                    metric.update(logits, labels_eval)

                    del images_eval, modal_xs_eval, labels_eval, logits

                    print_str = 'Eval {}/{}'.format(epoch, config.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch)

                    pbar.set_description(print_str, refresh=False)
                    pbar.update(1)

                # metric = evaluate_msf(model, val_loader,config, device,[1.0],False)
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                
            result_line = f'acc: {acc}, macc: {macc}, f1: {f1}, mf1: {mf1}, ious: {ious}, miou: {miou}\n'
            with open(config.log_dir + '/results.txt', 'a') as file:
                file.write(result_line)
            tb.add_scalar('val/macc', macc, epoch)
            tb.add_scalar('val/mf1', mf1, epoch)
            tb.add_scalar('val/miou', miou, epoch)

            if miou>best_miou:
                best_miou=miou
                torch.save(model.state_dict(), config.log_dir + '/checkpoint_epoch_{}_miou_{}.pth'.format(epoch, miou))
            print('miou',miou,'best',best_miou)

    # # TODO remove this back outside for loop
    # location = os.path.join(config.log_dir, "predictions")
    # pathlib.Path(location).mkdir(parents=True, exist_ok=True)
    # # Predict on the validation set
    # create_predictions(location, model, val_loader, config, device)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--channels', type=int, default=3)

    args = parser.parse_args()
    train(args.config, args)