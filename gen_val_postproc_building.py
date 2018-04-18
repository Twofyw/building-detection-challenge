from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import numpy as np
from numpy.random import randint
from random import random, uniform
import cv2 as cv

import argparse
import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import traceback

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels,unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian



def get_patch(sat_im, sat_size, stride, map_patch_size):
    '''
    Get small patches from the sat image. 
    
    sat_size - the length of sat image
    stride - the stride between two patches
    map_patch_size - useless for building
    '''
    h = sat_size 
    sat_im_pad = np.pad(sat_im, ((h,h),(h,h),(0,0)), 'reflect')
    
    for y in range(0, sat_im_pad.shape[0] + stride, stride):
        for x in range(0, sat_im_pad.shape[1] + stride, stride):
            if (y + sat_size) > sat_im_pad.shape[0]:
                continue
            if (x + sat_size) > sat_im_pad.shape[1]:
                continue

            sat_patch = np.copy(sat_im_pad[y:y + sat_size, x:x + sat_size, :])
            yield sat_patch/255.0, x, y


def pred_patches(model, sat_im, sat_patch_size, stride, batch, proc_method, map_patch_size):
    '''
    Predict patches 
    
    batch - batch size 
    proc_method - some function to process the prediction
    
    '''
    X = np.zeros((batch, sat_patch_size, sat_patch_size, 3), np.float32)
    i = 0
    for patch, x, y in get_patch(sat_im, sat_patch_size, stride, map_patch_size):
        X[i] = patch
        i += 1
        if i == batch:
            i = 0
            
            if not proc_method == None:
                yield map(lambda t: output_process(t, proc_method) ,model.predict_array(X))
            else:
                yield model.predict_array(np.moveaxis(X, -1, 1)).squeeze() ### Use your interface

def pred_one_map(model, sat_im, sat_patch_size=256, map_patch_size=256, stride=24, batch=1, proc_method=None, resize = 1, func=None):
    '''
    Predict one whole image 
    
    sat_patch_size - the size of small patch
    proc_method - some function to process the prediction
    resize - the ratio of resizing
    
    '''
    
    if not resize == 1: sat_im = cv.resize(sat_im, None, fx=resize, fy=resize)
        
    gen = pred_patches(model, sat_im, sat_patch_size, stride, batch, proc_method, map_patch_size)
    Y = next(gen)
    i = 0
    sat_size = sat_patch_size
    h = sat_patch_size
    map_pred = np.zeros((sat_im.shape[0], sat_im.shape[1]), np.float32)
    map_pred_pad = np.pad(map_pred, ((h,h),(h,h)), 'reflect')
    map_pred_deg_pad = np.copy(map_pred_pad)
    
    for y in range(0, map_pred_pad.shape[0] + stride, stride):
        for x in range(0, map_pred_pad.shape[1] + stride, stride):
            if (y + sat_size) > map_pred_pad.shape[0]:
                continue
            if (x + sat_size) > map_pred_pad.shape[1]:
                continue
                
            gain = int(sat_im.shape[0]/stride) ** 2  # To make the max value close to 1.0, Opt
        
            if func is not None:
                a = func(Y[i])
            else:
                a = Y[i]
            map_pred_pad[y+sat_patch_size//2-map_patch_size//2:y+sat_patch_size//2+map_patch_size//2,
                    x+sat_patch_size//2-map_patch_size//2:x+sat_patch_size//2+map_patch_size//2]\
                            += a * gain
            i += 1
            if i == batch:
                i = 0
                try: Y = next(gen)
                except StopIteration: 
                    return map_pred_pad
    return map_pred_pad


def CRF(img_, mask_, sxy, srgb):   ## Paras Need Tuneing
    mask = np.clip(mask_/255.,0,1)
    img = np.uint8(img_*255)
    mask = mask[np.newaxis, ...]
    mask = np.concatenate((1-mask, mask), 0)
    d = dcrf.DenseCRF2D(1024, 1024, 2)      # TO CHANGE: width, height, nlabels
    U = unary_from_softmax(mask); 
    d.setUnaryEnergy(U)
    
    d.addPairwiseGaussian(sxy=5, compat=3)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=img, compat=10)

    Q = d.inference(20)
    MAP = np.argmax(Q, axis=0); 
    MAP = MAP.reshape((1024,1024))         # TO CHANGE: width, height
    return MAP



def draw_poly(pred_map, overwrite = True, fill = False):
    pred_mask = np.uint8(pred_map>100)*255
    poly_im = np.copy(cv.cvtColor(pred_map_2,cv.COLOR_GRAY2RGB))
    poly_fill = np.zeros(pred_map.shape, np.uint8)
    m, contours, hierarchy = cv.findContours(
        pred_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    
    for c in contours:
        app_c = cv.approxPolyDP(c,5,True)
        last_point = (app_c[0,0,0], app_c[0,0,1])
        poly_im = cv.line(poly_im, last_point, (app_c[-1,0,0], app_c[-1,0,1]), (255,0,0), 2)
        for i, point in enumerate(app_c):
            ptr = (point[0,0],point[0,1])
            if not i==0:
                poly_im = cv.line(poly_im, ptr, last_point, (255,0,0), 2)
                last_point = ptr
    
        if fill:
            poly_fill = cv.fillPoly(poly_fill, app_c, 255)
            print(app_c)
    if fill: 
        return poly_fill
    return poly_im


def fit_poly(pred_map, thd):
    pred_mask = np.uint8(pred_map>thd)*255
    
    m, contours, hierarchy = cv.findContours(
        pred_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    
    polys = []
    for c in contours:
        app_c = cv.approxPolyDP(c,5,True)
        last_point = (app_c[0,0,0], app_c[0,0,1])
        #poly_im = cv.line(poly_im, last_point, (app_c[-1,0,0], app_c[-1,0,1]), (255,0,0), 2)
        points = []
        for i, point in enumerate(app_c):
            ptr = (point[0,0],point[0,1])
            points.append(ptr)
        
        polys.append(points)
    return polys


def draw_minRec(pred_map, overwrite = True):
    
    pred_mask = np.uint8(pred_map>80)*255
    poly_im = np.copy(cv.cvtColor(pred_map_2,cv.COLOR_GRAY2RGB))
    m, contours, hierarchy = cv.findContours(
        pred_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        rect = cv.minAreaRect(cnt)  
        box = cv.boxPoints(rect)  
        box = np.int0(box)  
        cv.drawContours(poly_im, [box], 0, (255, 0, 0), 2) 
        
    return poly_im


def fill_poly(pred_map):
    
    img = cv.compare(pred_map, 90, cv.CMP_GE)
    _, contours, hierarchy = cv.findContours(
        img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
    assert len(hierarchy) == 1 and len(contours) == len(hierarchy[0])
    polys = {}
    for i, (_, _, _, parent) in enumerate(hierarchy[0]):
        if cv.contourArea(contours[i]) <= 7:    #args.building_size:
            continue
        c = cv.approxPolyDP(contours[i], 5, True)
        if parent == -1:
            polys[i] = c, []
        else:
            polys[parent][1].append(c)
            
    mask = np.zeros(img.shape, np.uint8)
    for p in polys.values():
        cv.fillPoly(mask, [p[0]], 255)
    return mask


def post_proc(img):
    '''Post processing'''
    polys = fit_poly(img, 0.5)
    

    # Save polys to csv file...
    
    

def read_images(args):
    """Read the data and return the path of each image in a dict."""
    img_dir = args.val_dir
    print("Read images in " + img_dir)
    records = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file[-3:]=='png':
                records.append(dict({'image': os.path.join(img_dir, file)})) 
    
    print('%d images found!'%len(records))
    return records


def pred_images(model, args):
    ''' Predict images'''
    recodes = read_images(args)
    i = 0
    for rec in recodes:
        sat_fn = rec['image']
        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        if sat_im is None: 
            print('Error in reading: ',sat_fn)
        
        print('==== Reading ===',sat_fn)
        
        _, fn = os.path.split(sat_fn)
        map_fn = args.save_dir + fn[:-7] + 'mask.png'
        
        h = 256                   # the size of small patch

        pred_map = pred_one_map(model, sat_im, sat_patch_size = h, stride=args.stride, batch=args.batch, proc_method=args.proc_method, resize=args.resize, map_patch_size=args.map_size)
        
        
        pred_map = np.uint8(np.clip(pred_map, 0, 1)*255)
        pred_map = pred_map[h:-h, h:-h]
        
        pred_map = post_proc(pred_map)
        
        cv.imwrite(map_fn ,pred_map)
        print(i, map_fn)
        
        
