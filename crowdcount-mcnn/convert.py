import numpy as np

import os

import matplotlib.image as mpimg

import scipy.io as sio

from scipy.ndimage import filters

from sklearn.neighbors import NearestNeighbors

from PIL import Image

import math

from xmllist import read_content

def gaussian_filter_density(gt):

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))) #np.nonzero return two arrays, one is x_index of nonzero value and the other is y_index of nonzero value

    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)

    neighbors.fit(pts.copy())

    distances, _ = neighbors.kneighbors()

    density = np.zeros(gt.shape, dtype=np.float32)

    type(distances)

    sigmas = distances.sum(axis=1) * 0.075 # 0.075 = 0.3/4

    for i in range(len(pts)):

        pt = pts[i]

        pt2d = np.zeros(shape=gt.shape, dtype=np.float32)

        pt2d[pt[1]][pt[0]] = 1

        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')

    return density



def create_density(gts, d_map_h, d_map_w):

    res = np.zeros(shape=[d_map_h, d_map_w]) # res store head positions

    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)

    for k in range(len(gts)):

        gt = gts[k]

        if (bool_res[k] == True):

            res[int(gt[1])][int(gt[0])] = 1

    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))

    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)

    neighbors.fit(pts.copy())

    distances, _ = neighbors.kneighbors()

    map_shape = [d_map_h, d_map_w]

    density = np.zeros(shape=map_shape, dtype=np.float32)

    sigmas = distances.sum(axis=1) * 0.075

    for i in range(len(pts)):

        pt = pts[i]

        pt2d = np.zeros(shape=map_shape, dtype=np.float32)

        pt2d[pt[1]][pt[0]] = 1

        t1 = filters.gaussian_filter(pt2d, sigmas[i])

        density += t1 #bigger the sigma, blurrer the picture

    return density



if __name__ == '__main__':

    train_img = '/home/bedant/crowdcount-mcnn/eval/'

    out_path = '/home/bedant/crowdcount-mcnn/output/'


    img_names = os.listdir(train_img)

    num = len(img_names)

    annPoints = read_content("/home/bedant/crowdcount-mcnn/annotations.xml")

    global_step = 1

    for i in range(num):

        key, gts = next(iter(annPoints.items()))
        full_img = train_img + key
        gts = np.array(gts)

        img = mpimg.imread(full_img)
        # shape is (num_count, 2)


        shape = img.shape

        if (len(shape) < 3):

            img = img.reshape([shape[0], shape[1], 1])


        d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)

        d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)

        den_map = create_density(gts / 4, d_map_h, d_map_w)



        p_h = math.floor(float(img.shape[0]) / 3.0)

        p_w = math.floor(float(img.shape[1]) / 3.0)

        d_map_ph = math.floor(math.floor(p_h / 2.0) / 2.0)

        d_map_pw = math.floor(math.floor(p_w / 2.0) / 2.0)


        py = 1

        py2 = 1

        for j in range(1, 4):

            px = 1

            px2 = 1

            for k in range(1, 4):

                final_image = img[py - 1: py + p_h - 1, px - 1: px + p_w - 1, :]

                final_gt = den_map[py2 - 1: py2 + d_map_ph - 1, px2 - 1: px2 + d_map_pw - 1]

                px = px + p_w

                px2 = px2 + d_map_pw

                if final_image.shape[2] < 3:

                    final_image = np.tile(final_image, [1, 1, 3]) #tile : repeat "final_image" once in axis=0 and axis = 1 while repeat three times in axis=2

                image_final_name = out_path + 'img/' + key

                gt_final_name = out_path  + 'gt/' + 'GT_' + key.split('.')[0]

                Image.fromarray(final_image).convert('RGB').save(image_final_name)

                np.save(gt_final_name, final_gt)



            py = py + p_h

            py2 = py2 + d_map_ph
