import numpy as np
import cv2
from scipy import spatial
import matplotlib.pyplot as plt
from PIL import Image


def X2Cube(img):

    B = [4, 4]
    skip = [4, 4]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    # print ('out.shape = ',out.shape)
    # print ('M = ',M,' , N = ',N)
    img = out.reshape(M//4, N//4, 16)
    # img = img[:,:,0:15]
    # print ('img.shape = ',img.shape)
    #img = img.transpose(1,0,2)
    #print ('---222---img.shape = ',img.shape)
    #img = img / img.max() * 255 #  归一化
    img.astype('uint8')
    return img


def similarity_checking(a_img, b_img):
    a_array = a_img.flatten()
    b_array = b_img.flatten()

    similarity = -1 * (spatial.distance.cosine(a_array / 255, b_array / 255) - 1)
    return similarity

def background_aware_band_selection(img, gt):
    f_img = img[gt[1]:gt[1] + gt[3], gt[0]:gt[0] + gt[2], :]
    padding_gt = gt.copy()

    padding = 50
    padding_gt[0] = padding_gt[0] - int(padding / 2)
    padding_gt[1] = padding_gt[1] - int(padding / 2)
    padding_gt[2] = padding_gt[2] + padding
    padding_gt[3] = padding_gt[3] + padding

    if padding_gt[0] < 0:
        padding_gt[2] = padding_gt[2] - padding_gt[0]
        padding_gt[0] = 0
    if padding_gt[1] < 0:
        padding_gt[3] = padding_gt[3] - padding_gt[1]
        padding_gt[1] = 0
    if padding_gt[0] + padding_gt[2] > img.shape[1]:
        padding_gt[2] = img.shape[1] - padding_gt[0]
    if padding_gt[1] + padding_gt[3] > img.shape[0]:
        padding_gt[3] = img.shape[0] - padding_gt[1]

    t_img = img.copy()
    t_img[gt[1]:gt[1] + gt[3], gt[0]:gt[0] + gt[2], :] = 0
    b_img = t_img[padding_gt[1]:padding_gt[1] + padding_gt[3], padding_gt[0]:padding_gt[0] + padding_gt[2], :]

    sum_of_similarity = np.zeros(img.shape[2])

    for i in range(img.shape[2]):
        for j in range(i + 1, img.shape[2]):
            obejct_similarity = similarity_checking(f_img[:, :, i], f_img[:, :, j])
            background_similarity = similarity_checking(b_img[:, :, i], b_img[:, :, j])
            difference = abs(background_similarity - obejct_similarity)
            sum_of_similarity[i] = sum_of_similarity[i] + difference
            sum_of_similarity[j] = sum_of_similarity[j] + difference

    temp_value = np.flip(np.argsort(sum_of_similarity))
    h, w = img[:, :, 0].shape
    image = np.zeros((h, w, 3))
    i=0
    image[:, :, 0] = img[:, :, temp_value[i * 3]]
    image[:, :, 1] = img[:, :, temp_value[i * 3 + 1]]
    image[:, :, 2] = img[:, :, temp_value[i * 3 + 2]]
    image = image / image.max() * 255
    image = np.uint8(image)

    return image, temp_value
