#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
import cv2
import numpy as np
import cv2.aruco as aruco

def aruco_detection(frame, aruco_detector):
    """
    use aruco_detector to detect quads in frame
    1. enhance red channel and subtract blue channel to get a gray image
    2. apply aruco_detector.detectMarkers to the gray image to get quads
    return:
    `quads_aruco`: list of quad (np.array with shape (1, 4, 2)), counter-clockwise
    """
    ## split channels
    B, G, R = cv2.split(frame.astype(np.int16))
    ## red enhancement
    R += ( 1.5 * np.clip(R - G, 0, 255) ).astype(np.int16)
    grayImg = np.clip(R - B, 0, 255).astype(np.uint8)

    # cv2.imshow("frame enhanced", frame_enhanced)
    # cv2.imshow("grayImg", grayImg)

    quads_aruco, _, _ = aruco_detector.detectMarkers(cv2.bitwise_not(grayImg))
    ## convert to counter-clockwise (actually for warping)
    quads_aruco = [quad[:, ::-1, :] for quad in quads_aruco]
    
    ## show quads in enhanced frame
    # if len(quads_aruco) > 0:
    #     print("--------------------\naruco find quads")
    # frame_enhanced = frame.copy()
    # frame_enhanced[:, :, 2] = np.clip(R, 0, 255).astype(np.uint8)
    # frame_aruco = cv2.drawContours(frame_enhanced, np.array(quads_aruco, dtype=int), -1, (0, 255, 0), 2)
    # cv2.imshow("frame_aruco", frame_aruco)

    return quads_aruco

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


from simple_digits_classification.simple_digits_classify import CNN_digits
import torch
from torch import nn
import torch.nn.functional as F

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
model = CNN_digits(C_in=3, H_in=50, W_in=50, n_classes=9)
model.load_state_dict(torch.load(dir_path + "/simple_digits_classification/model.pth"))
model.eval()

def classification_cnn(frame_cv, quads):
    """
    use cnn to classify the digit, works better than template matching
    Note: the input frame_cv is in opencv format
        opencv format: 0-255, (h, w, 3), BGR
        torch format: 0.0-1.0, (3, H_in, W_in), RGB
    """
    global model
    h, w = model.H_in, model.W_in

    quads_id = []
    if len(quads) == 0:
        return quads_id

    ## convert opencv image format to "torch format" (consistent with how transform.Compose() does in traing)
    frame = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 256.0
    dst = np.array(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32
    )
    warped_list = []
    for i in range(len(quads)):
        src = quads[i].astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)  # 获取变换矩阵
        warped = cv2.warpPerspective(frame, M, (h, w))  # 进行变换
        warped_list.append(warped)
    X = np.stack(warped_list, axis=0) 
    ## (B, H_in, W_in, 3) => (B, 3, H_in, W_in)
    X = torch.tensor(X, dtype=torch.float).permute(0, 3, 1, 2)
    ## get logits
    logits_list = model(X).detach()
    for i, logits in enumerate(logits_list):
        ## pairs: (id, score)
        pairs = [(i+1, round(logit.item(), 1)) for i, logit in enumerate(logits.reshape(-1))]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        if pairs[0][1] - pairs[1][1] > 15:
            quad_id = pairs[0][0] + 1
        else:
            quad_id = 0
        quads_id.append(quad_id)
        ########## for debug ##########
        cv2.imshow(f"id: {pairs[0][0]}, margin: {pairs[0][1] - pairs[1][1]:.1g}", cv2.cvtColor(warped_list[i], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        print(pairs)
        ########## debug end ##########
    return quads_id

def get_custom_dict():
    custom_dict = aruco.Dictionary()
    custom_dict.markerSize = 5
    custom_dict.maxCorrectionBits = 15
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
    global aruco_detector
    quads_f = aruco_detection(frame, aruco_detector)
    quads, tvec_list, rvec_list, area_list, _ = quads_reconstruction(quads_f, camera_matrix, height_range=height_range)
    """ simple cnn classifier """

    # import time
    # sta = time.time()
    quads_id = classification_cnn(frame, quads)
    # print(f"cost: {time.time() - sta:.3f} sec")

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

def test():
    """ read image from rgb.avi and test marker_detection """
    cap = cv2.VideoCapture("./rgb.avi")
    cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        cnt += 1
        if not ret:
            break
        if cnt < 800:
            continue
        cv2.imshow("frame", frame)
        camera_matrix = np.array(
           [[607.5924072265625, 0.0, 426.4002685546875],
            [0.0, 606.0050048828125, 242.9524383544922],
            [0.0, 0.0, 1.0]]).reshape((3, 3))
        marker_detection(frame, camera_matrix)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
        # print(f"frame {cnt}")

if __name__ == '__main__':
    test()