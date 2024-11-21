import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np

import torch

sys.path.insert(0, '.')
from data_prov import RegionDataset
from modules.model import MDNet, set_optimizer, BCELoss, Precision



def get_GT_Bbox(gt_fileName):
    f = open(gt_fileName, 'r')
    lines = f.readlines()
    f.close()
    gt_list = []
    # print(gt_fileName)
    for line in lines:
        kk = line.split('\t')[:-1]
        # print(kk)
        gt_list.append(list(map(int, kk)))
    return gt_list


def get_Image_data(img_directory):
    img_list = []
    img_list_temp = os.listdir(img_directory)
    img_list_temp.sort()
    for img_path in img_list_temp:
        if img_path.find('.png') != -1:
            img_list.append(os.path.join(img_directory, img_path))
    return img_list


def getSourceData(video_path):
    # print('video_path = ', video_path)
    img_list = get_Image_data(os.path.join(video_path, 'HSI'))

    gt_list = get_GT_Bbox(os.path.join(video_path, 'HSI', 'groundtruth_rect.txt'))
    assert len(img_list) == len(gt_list)
    return img_list, gt_list


def generate_image_gt_list(rootDir):
    final_img_list = []
    final_gt_list = []
    for video_name in os.listdir(rootDir):
        img_list, gt_list = getSourceData(os.path.join(rootDir, video_name))
        final_img_list.append(img_list)
        final_gt_list.append(gt_list)

    return final_img_list, final_gt_list


def train_mdnet(opts, train_dataset_dir=''):
    # Init dataset
    final_img_list, final_gt_list = generate_image_gt_list(rootDir=train_dataset_dir)
    K = len(final_img_list)
    print('K = ', K)
    dataset = [None] * K
    for k in range(K):
        dataset[k] = RegionDataset(final_img_list[k], final_gt_list[k], opts)
        # Init model
    model = MDNet(opts['model_path'], K)
    print(model)
    for param in model.parameters():
        param.requires_grad = False
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    for name, param in model.named_parameters():
        if param.requires_grad:
            print('trained name = ', name)

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])
    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        for j, k in enumerate(k_list):
            #print(k)
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next()
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            #print(len(pos_regions))
            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            loss = criterion(pos_score, neg_score)

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                  .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))

        print('Mean Precision: {:.3f}'.format(prec.mean()))
        print('Save model to {:s}'.format(opts['model_path']))
        if opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='D:\\Dataset_HSI_Tracking\\Train',
                        help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open('D:\\phd_code\\Background_Aware_Band_Selection\\pretrain\\options_vot.yaml', 'r'))
    train_mdnet(opts, args.dataset)
