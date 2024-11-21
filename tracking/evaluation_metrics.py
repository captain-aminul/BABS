import numpy as np


def cal_iou(box1, box2):
    """

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou


def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)


def calAUC(gtArr,resArr, video_dir):
    # ------------ starting evaluation  -----------
    success_all_video = []
    for idx in range(len(resArr)):
        result_boxes = resArr[idx]
        result_boxes_gt = gtArr[idx]
        result_boxes_gt = [np.array(box) for box in result_boxes_gt]
        iou = list(map(cal_iou, result_boxes, result_boxes_gt))
        success = cal_success(iou)
        auc = np.mean(success)
        success_all_video.append(success)
        # print ('video = ',video_dir[idx],' , auc = ',auc)
    return np.mean(success_all_video)