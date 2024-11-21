import math

import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.utils.data as data
import torch.optim as optim

from tracking.evaluation_metrics import calAUC

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.band_selection import X2Cube, background_aware_band_selection

from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
from tracking.gen_config import gen_config

sys.path.insert(0,'./gnet')
from gnet.g_init import NetG, set_optimizer_g
from gnet.g_pretrain import *


def display_image(img, gt, title=''):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=1, facecolor='none', edgecolor='r'))
    ax.set_title(title)
    plt.show()

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def train(model, model_g, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        if model_g is not None:
            batch_asdn_feats = pos_feats.index_select(0, pos_cur_idx)
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        if model_g is not None:
            model_g.eval()
            res_asdn = model_g(batch_asdn_feats)
            model_g.train()
            num = res_asdn.size(0)
            mask_asdn = torch.ones(num, 512, 3, 3)
            res_asdn = res_asdn.view(num, 3, 3)
            for i in range(num):
                feat_ = res_asdn[i, :, :]
                featlist = feat_.view(1, 9).squeeze()
                feat_list = featlist.detach().cpu().numpy()
                idlist = feat_list.argsort()
                idxlist = idlist[:3]

                for k in range(len(idxlist)):
                    idx = idxlist[k]
                    row = idx // 3
                    col = idx % 3
                    mask_asdn[:, :, col, row] = 0
            mask_asdn = mask_asdn.view(mask_asdn.size(0), -1)
            if opts['use_gpu']:
                batch_asdn_feats = batch_asdn_feats.cuda()
                mask_asdn = mask_asdn.cuda()
            batch_asdn_feats = batch_asdn_feats * mask_asdn

        # forward
        if model_g is None:
            pos_score = model(batch_pos_feats, in_layer=in_layer)
        else:
            pos_score = model(batch_asdn_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if model_g is not None:
            start = time.time()
            prob_k = torch.zeros(9)
            for k in range(9):
                row = k // 3
                col = k % 3

                model.eval()
                batch = batch_pos_feats.view(batch_pos, 512, 3, 3)
                batch[:, :, col, row] = 0
                batch = batch.view(batch.size(0), -1)

                if opts['use_gpu']:
                    batch = batch.cuda()

                prob = model(batch, in_layer='fc4', out_layer='fc6_softmax')[:, 1]
                model.train()

                prob_k[k] = prob.sum()

            _, idx = torch.min(prob_k, 0)
            idx = idx.item()
            row = idx // 3
            col = idx % 3

            optimizer_g = set_optimizer_g(model_g)
            labels = torch.ones(batch_pos, 1, 3, 3)
            labels[:, :, col, row] = 0

            batch_pos_feats = batch_pos_feats.view(batch_pos_feats.size(0), -1)
            res = model_g(batch_pos_feats)
            labels = labels.view(batch_pos, -1)
            criterion_g = torch.nn.MSELoss(reduction='mean')
            loss_g_2 = criterion_g(res.float(), labels.cuda().float())
            model_g.zero_grad()
            loss_g_2.backward()
            optimizer_g.step()

            end = time.time()
            # print('asdn objective %.3f, %.2f s' % (loss_g_2, end - start))



def save_image(image, predicted_gt, original_gt, count=1):
    fig = plt.figure(frameon=False, figsize=(image.size[0] / 80, image.size[1] / 80), dpi=80)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    ax.add_patch(patches.Rectangle((original_gt[0], original_gt[1]), original_gt[2], original_gt[3],
                                   linewidth=3, facecolor='none', edgecolor='g'))
    ax.add_patch(patches.Rectangle((predicted_gt[0], predicted_gt[1]), predicted_gt[2],
                                   predicted_gt[3], linewidth=3, facecolor='none', edgecolor='r'))
    fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(count)), dpi=80.0)

    plt.close()


def hsi_band_grouping(img, band_order):
    results = []
    h, w = img[:, :, 0].shape
    best_index = -1
    value = -99999999
    for i in range(int(img.shape[2]/3)):
        image = np.zeros((h, w, 3))
        image[:, :, 0] = img[:, :, band_order[i*3]]
        image[:, :, 1] = img[:, :, band_order[i*3 + 1]]
        image[:, :, 2] = img[:, :, band_order[i*3 + 2]]
        image = image / image.max() * 255
        image = np.uint8(image)
        results.append(image)

    return results

def generate_bboxes_nearby(original_bbox, image_size, n=500, target_iou=0.7, iou_tolerance=0.05):
    generated_bboxes = []
    image_w, image_h = image_size
    original_bbox = np.array(original_bbox)

    while len(generated_bboxes) < n:
        # Generate random perturbations to the original bbox
        new_bbox = original_bbox.copy()

        # Random translation (shift)
        new_bbox[0] += np.random.uniform(-0.1, 0.1) * original_bbox[2]  # Shift x
        new_bbox[1] += np.random.uniform(-0.1, 0.1) * original_bbox[3]  # Shift y

        # Random scale (size change)
        new_bbox[2] *= np.random.uniform(0.9, 1.1)  # Scale width
        new_bbox[3] *= np.random.uniform(0.9, 1.1)  # Scale height

        # Ensure the bbox stays within image bounds
        new_bbox[0] = np.clip(new_bbox[0], 0, image_w - new_bbox[2])
        new_bbox[1] = np.clip(new_bbox[1], 0, image_h - new_bbox[3])
        new_bbox[2] = np.clip(new_bbox[2], 1, image_w - new_bbox[0])  # Ensure width is positive
        new_bbox[3] = np.clip(new_bbox[3], 1, image_h - new_bbox[1])  # Ensure height is positive

        # Calculate IoU with the original bounding box
        iou = overlap_ratio(new_bbox, original_bbox)

        # Check if the IoU is within the desired range
        if target_iou - iou_tolerance <= iou <= target_iou + iou_tolerance:
            generated_bboxes.append(new_bbox)

    return np.array(generated_bboxes)

def run_vital(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    # Init bbox
    target_bbox = np.array(init_bbox)

    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox
    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    #print(opts['model_path'])
    model = MDNet(opts['model_path'])
    #print(model)
    model_g = NetG()
    if opts['use_gpu']:
        model = model.cuda()
        model_g = model_g.cuda()

    # Init criterion and optimizer
    criterion = BCELoss()
    criterion_g = torch.nn.MSELoss(reduction='mean')

    model.set_learnable_params(opts['ft_layers'])
    model_g.set_learnable_params(opts['ft_layers'])

    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image1 = Image.open(img_list[0])
    image = np.array(image1)
    image = X2Cube(image)
    h, w = image.shape[:2]
    image, band_order = background_aware_band_selection(image, target_bbox)
    t_img = Image.fromarray(image)
    popo_img = image
    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', t_img.size, opts['trans_pos'], opts['scale_pos'])(
        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        SampleGenerator('uniform', t_img.size, opts['trans_neg_init'], opts['scale_neg_init'])(
            target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
        SampleGenerator('whole', t_img.size)(
            target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])

    neg_examples = np.random.permutation(neg_examples)
    # Extract pos/neg features
    if len(pos_examples) == 0:
        pos_examples = generate_bboxes_nearby(target_bbox, t_img.size)

    pos_feats = forward_samples(model, popo_img, pos_examples)
    neg_feats = forward_samples(model, popo_img, neg_examples)

    # Initial training
    train(model, None, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

    del init_optimizer, neg_feats
    torch.cuda.empty_cache()
    g_pretrain(model, model_g, criterion_g, pos_feats)
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', t_img.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                     opts['aspect_bbreg'])(
        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, popo_img, bbreg_examples)
    t_bbreg_examples = []

    bbreg = BBRegressor(t_img.size)
    # print(len(bbreg_examples))
    # print(len(bbreg_feats))
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', t_img.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', t_img.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', t_img.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, popo_img, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (t_img.size[0] / dpi, t_img.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(t_img, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle((gt[0][0], gt[0][1]), gt[0][2], gt[0][3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        # if display:
        #     plt.pause(.01)
        #     plt.draw()
        # if savefig:
        #     fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list)):
        tic = time.time()
        image = Image.open(img_list[i])
        image = np.array(image)
        image = X2Cube(image)

        images = hsi_band_grouping(image, band_order)



        #image = images[0]
        t_img = Image.fromarray(images[0])
        image = images[0]

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, None, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, model_g, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf



        # Display
        # if display or savefig:
        #     #print(result_bb[i], gt[i])
        #     save_image(t_img, result_bb[i], gt[i], count=i)

        if gt is None:
            pass
            #print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
             #     .format(i + 1, len(img_list), target_score, spf))
        else:
            #print(f'original gt: {gt[i]}, result_bb[i]:{result_bb}')
            overlap[i] = overlap_ratio(np.array(gt[i]), result_bb[i])[0]
            #print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                  # .format(i + 1, len(img_list), overlap[i], target_score, spf))
        # print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
        # .format(i + 1, len(img_list), overlap[i], target_score, spf))
    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    return result, result_bb



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--seq', default='D:\\hsi_2023_dataset\\validation\hsi\\nir', help='input seq')
    parser.add_argument('-s', '--seq', default='D:\\HOT_2022\\Test', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    assert args.seq != '' or args.json != ''

    video_dir_arr = os.listdir(args.seq)
    video_dir_arr.sort()

    for video in video_dir_arr:
        np.random.seed(0)
        torch.manual_seed(0)
        print(f'current video name: {video}')
        img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args, video)
        # Run tracker
        result, result_bb =  run_vital(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)
        result_path = "../Results/HOT_2022"
        filename = os.path.join(result_path, f'{video}.txt')
        auc = calAUC([gt], [result_bb], [f"{video}"])
        print(video, "AUC:", auc)

        if os.path.exists(filename):
            with open(filename, 'w') as file:
                for box in result_bb:
                    for bb in box:
                        file.write(str(bb) + "\t")
                    file.write("\n")
            file.close()
        else:

            with open(filename, 'x') as file:
                for box in result_bb:
                    for bb in box:
                        file.write(str(bb) + "\t")
                    file.write("\n")
            file.close()

