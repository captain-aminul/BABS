import os
import json
import numpy as np

# D:\phd_code\Background_Aware_Band_Selection\Results\single_image\ball\result
def gen_config(args, video_name):
    if args.seq != '':
        # generate config from a sequence name
        # part1: band selection for every frame
        # part2: band selection only once

        seq_home = args.seq
        result_home = 'Results/HOT_2024'

        seq_name = video_name
        img_dir = os.path.join(seq_home, seq_name, 'HSI')
        # img_dir = os.path.join(seq_home, seq_name)
        img_list_temp = os.listdir(img_dir)
        img_list_temp.sort()
        img_list = []

        for img in img_list_temp:
            if(img.find('.png')!=-1):
                img = os.path.join(img_dir, img)
                img_list.append(img)


        gt_path = os.path.join(seq_home, seq_name, 'HSI', 'groundtruth_rect.txt')
        # gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
        f = open(gt_path, 'r')
        lines = f.readlines()
        f.close()
        gt = []

        for line in lines:
            gt_data_per_image = line.split('\t')[:-1]
            if len(gt_data_per_image)==3:
                gt_data_per_image = line.split('\t')
            gt_data_int = list(map(int, gt_data_per_image))
            #print(gt_data_int)
            gt.append(gt_data_int)

        gt = np.asarray(gt)
        init_bbox = gt[0]

        result_dir = os.path.join(result_home, seq_name)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, 'figs')
        result_path = os.path.join(result_dir, 'result')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    # if savefig_dir!='':
    #    if not os.path.exists(savefig_dir):
    #        os.makedirs(savefig_dir)
    # else:
    #    savefig_dir = ''

    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path