#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
import cv2
import numpy as np
import cv2.aruco as aruco

def preprocessing(frame):
    """
    Processing the image to get the binary image with HSV red filter
    Note:
        Be careful while modifying HSV filter range, you may
        need to save warped images to the training set again
    """
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boolImg = (
        np.logical_and(
            np.logical_and(
                np.logical_or(hsvImg[:, :, 0] <= 10, hsvImg[:, :, 0] >= 150),
                hsvImg[:, :, 1] >= 130,
            ),
            hsvImg[:, :, 2] >= 70,
        )
        * 255
    ).astype(np.uint8)
    # boolImg = (np.logical_and(np.logical_and(np.logical_or(hsvImg[:,:,0] <= 10 , hsvImg[:,:,0] >= 150) , hsvImg[:,:,1] >= 100) , hsvImg[:,:,2] >= 50) * 255).astype(np.uint8)
    return boolImg, hsvImg


from simple_digits_classification.simple_digits_classify import CNN_digits
import torch
from torch import nn
import torch.nn.functional as F

file_path = os.path.dirname(__file__)
model = CNN_digits(50, 50, 9)
model.load_state_dict(torch.load(file_path + "/simple_digits_classification/model.pth"))
model.eval()


def classify(image, is_white_digit=True):
    """
    Input:
        `image`: grayscale image. (h, w)
        `is_white_digit`: if the digit is white on black background, set it to True, otherwise False
    Output:
        `idx`: classified result. [0, 1, 2, 3, 4, 5] => block 1-6; 6-8 => block B, O, X; -1: unknown
        `logits`: raw output of the model, softmax(logits) is the probability of each class
    TODO: find more 'unknown' cases and add them to the training set
    """
    global model
    # if black digit on white background, invert the image
    if not is_white_digit:
        image = cv2.bitwise_not(image)

    image = cv2.resize(image, (50, 50))
    x = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
    logits = model(x).detach()
    idx = torch.argmax(logits, dim=1).item()
    return idx, logits

def quads_detection(grayImg, area_thresh=225):
    """
    Detect quads (warped block surfaces) in grayImg
    
    `area_thresh`: threshold to filter out small contours, like noise and quads far away
        set it to 50 => can detect quads 2m away but sometimes may confuse with "6" and "B"
        set it to 225 => can only detect quads 1.5m away but detected quads are bigger and more clear,
        for which classifier works better (nearly 100% acc).
    """
    quads_f = []

    contours, hierarchy = cv2.findContours(
        grayImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_filt = []
    for i, c in enumerate(contours):
        child_idx = hierarchy[0, i, 2]
        ## must have digits inside
        if child_idx == -1:  # or cv2.contourArea(contours[child_idx]) < 10:
            continue
        if cv2.contourArea(c) < area_thresh:
            continue
        contours_filt.append(c)
        # print("contour added")
        # cv2.drawContours(rgbImage, [c], -1, (255, 0, 0), 3)
        # cv2.imshow("rgbImage", rgbImage)
        # cv2.waitKey(0)

    quads_aruco, _, _ = aruco_detector.detectMarkers(cv2.bitwise_not(grayImg))
    quads_aruco = np.squeeze(quads_aruco)
    if len(quads_aruco) == 0:
        return []

    for i, contour in enumerate(contours_filt):
        x, y, w, h = cv2.boundingRect(contour)
        if h / w > 1.8 or w / h > 1.8:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        """ aruco neighbour filter """
        ## quads_aruco: (n, 4, 2), approx: (4, 1, 2)
        approx = np.squeeze(approx)
        diffs = np.squeeze(quads_aruco - approx) ## diff shape of (n, 4, 2)
        assert diffs.shape[-2:] == (4, 2)
        cent_diffs = np.mean(diffs, axis=1)
        cent_dists = [np.linalg.norm(cent_diff) for cent_diff in cent_diffs]
        if min(cent_dists) > 10:     ## min distance >= 5 pixels
            # print("aruco filter out a quad.")
            continue
        """ warped rect found """
        quads_f.append(approx.astype(float))
    return quads_f

def quads_reconstruction(quads, camera_matrix, height_range=(-10.0, 10.0)):
    """
    Reconstruct blocks' poses relative to camera frame by pnp solving of quads
    `quads`: list of quad (np.array with shape (1, 4, 2)), with dtype of float or int
        e.g. return value of cv2.approxPolyDP or cv2.aruco.ArucoDetector.detectMarkers
    `camera_matrix`: used to solve pnp
    `height_range`: used to distinguish between block1-6 (no higher than +0.2m,
        much lower when not stacked) and gameinfo board (much higher than blocks)
        note that y axis in camera frame is downward.
        e.g. (-0.2, 1.0) => blocks, (-10.0, -0.2) => gameinfo board
    """
    rvec_list = []
    tvec_list = []
    quads_prj = []
    area_list = []
    indices = []

    block_size = 0.045
    """ block center's (0, 0, 0) """
    half_len = 0.5 * block_size
    model_object = np.array(
        [
            (-half_len, -half_len, -half_len),
            (-half_len, +half_len, -half_len),
            (+half_len, +half_len, -half_len),
            (+half_len, -half_len, -half_len),
        ]
    )
    distort_coeffs = np.array([[0, 0, 0, 0]], dtype=np.float32)
    for idx, quad in enumerate(quads):
        model_image = np.squeeze(quad).astype(np.float32)
        """ calculate the pose of the corner points by pnp solving """
        ret, rvec, tvec = cv2.solvePnP(
            model_object, model_image, camera_matrix, distort_coeffs
        )
        """ reconstruct corner points in image plane """
        projectedPoints, _ = cv2.projectPoints(
            model_object, rvec, tvec, camera_matrix, distort_coeffs
        )

        err = 0
        """ compute the reconstruction error """
        for t in range(len(projectedPoints)):
            err += np.linalg.norm(projectedPoints[t] - model_image[t])
        area = cv2.contourArea(quad.astype(np.int32))
        """ check if reconstruction error is small enough """
        if err / area > 0.005:
            continue

        quad_height = tvec[1]
        # print(f"quad: {quad}")
        """ chech if quad_height is within given height_range """
        if quad_height < height_range[0] or quad_height > height_range[1]:
            continue

        quads_prj.append(np.round(projectedPoints).astype(int))
        rvec_list.append(rvec)
        tvec_list.append(tvec)
        area_list.append(area)
        indices.append(idx)
    return quads_prj, tvec_list, rvec_list, area_list, indices

def classification_cnn(grayImg, quads):
    """ use cnn to classify the digit, works better than template matching """
    quads_id = []
    warped_img_list = []

    for i in range(len(quads)):
        src = quads[i].astype(np.float32)
        l = 50
        dst = np.array(
            [[0, 0], [0, l - 1], [l - 1, l - 1], [l - 1, 0]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(src, dst)  # 获取变换矩阵
        warped = cv2.warpPerspective(grayImg, M, (l, l))  # 进行变换
        ## save warped image
        warped_img_list.append(warped)
        idx, logits = classify(warped, is_white_digit=False)
        pairs = [(i+1, logit) for i, logit in enumerate(logits.reshape(-1))]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        if pairs[0][1] - pairs[1][1] > 1000:
            quad_id = idx + 1
        else:
            quad_id = 0
        quads_id.append(quad_id)
        ########## for debug ##########
        # cv2.imshow(f"warped {i}", warped)
        # cv2.waitKey(0)
        # print(pairs)
        ########## debug end ##########
    return quads_id, warped_img_list

def get_custom_dict():
    custom_dict = aruco.Dictionary()
    custom_dict.markerSize = 5
    custom_dict.maxCorrectionBits = 8
    markar_byte_list = []

    marker_bits = np.array(
        [
            # num 1
            [
                [0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            # num 2
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            # num 3
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            # num 4
            [
                [0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0],
            ],
            # num 5
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
            ],
            # num 6
            [
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            # char B
            [
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
            ],
            # char O
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            # char X
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
            ],
        ],
        dtype=np.uint8,
    )

    for i in range(len(marker_bits)):
        compressed = aruco.Dictionary.getByteListFromBits(marker_bits[i])
        markar_byte_list.append(compressed[0])
    byte_list = np.array(markar_byte_list)
    custom_dict.bytesList = byte_list
    return custom_dict

custom_dict = get_custom_dict()
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(custom_dict, aruco_params)
def marker_detection(
    frame,
    camera_matrix,
    verbose=True,
    height_range=(-10, 10)
):
    """
    detect markers and poses of quads in RGB image `frame`
    Input:
        `height_range`: used to distinguish between block1-6 (no higher than +0.2m,
            much lower when not stacked) and gameinfo board (much higher than blocks).
            note that y axis in camera frame is downward.
            e.g. (-0.2, 1.0) => blocks, (-10.0, -0.2) => gameinfo board.
    Output:
        quads_id, quads, area_list, tvec_list, rvec_list
    """
    boolImg, _ = preprocessing(frame)

    """ opencv morphological detector """
    quads_f = quads_detection(boolImg, area_thresh=300)
    quads, tvec_list, rvec_list, area_list, _ = quads_reconstruction(quads_f, camera_matrix, height_range=height_range)
    """ simple cnn classifier """
    quads_id, _ = classification_cnn(boolImg, quads)
    if verbose:
        # print(f"detected: {quads_id}")
        id2tag = {
            0: "*",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "B",
            8: "O",
            9: "X",
        }
        for i in range(len(quads)):
            if quads_id[i] == 0:
                continue
            bbox = cv2.boundingRect(quads[i])
            try:
                cv2.putText(
                    frame,
                    id2tag[quads_id[i]],
                    (bbox[0], bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            except:
                traceback.print_exc()
        cv2.drawContours(frame, quads, -1, (0, 255, 0), 1)
    # extract indices of valid quads in `quads_ID`
    ids = [i for i in range(len(quads_id)) if quads_id[i] != 0]
    return (
        [quads_id[_] for _ in ids],
        [quads[_] for _ in ids],
        [area_list[_] for _ in ids],
        [tvec_list[_] for _ in ids],
        [rvec_list[_] for _ in ids],
    )
