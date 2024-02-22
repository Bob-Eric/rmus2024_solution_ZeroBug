#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
import cv2
import copy
import rospy
import numpy as np

templates = []


def map_img77(img):
    segment = [
        6,
        14,
        22,
        30,
        37,
        44,
    ]
    ass = np.split(img, segment, axis=0)
    all_subs = np.array(
        [
            [np.sum(k) / k.size / 255.0 for k in np.split(a, segment, axis=1)]
            for a in ass
        ],
        dtype=np.float,
    )
    return (all_subs > 0.5).astype(np.uint8) * 255


def load_template():
    global templates
    tpl_path = os.path.join(os.path.dirname(__file__), "template/")
    rospy.loginfo(tpl_path)
    for i in range(1, 9):
        tpl = cv2.imread(tpl_path + str(i) + ".png", 0)
        rospy.loginfo(tpl_path + str(i) + ".png")
        rospy.loginfo(tpl.shape)
        templates.append(map_img77(tpl))


def sort_contour(cnt):

    if not len(cnt) == 4:
        assert False
    new_cnt = cnt.copy()

    cx = (cnt[0, 0, 0] + cnt[1, 0, 0] + cnt[2, 0, 0] + cnt[3, 0, 0]) / 4.0
    cy = (cnt[0, 0, 1] + cnt[1, 0, 1] + cnt[2, 0, 1] + cnt[3, 0, 1]) / 4.0

    x_left_n = 0
    for i in range(4):
        if cnt[i, 0, 0] < cx:
            x_left_n += 1
    if x_left_n != 2:
        return None
    lefts = np.array([c for c in cnt if c[0, 0] < cx])
    rights = np.array([c for c in cnt if c[0, 0] >= cx])
    if lefts[0, 0, 1] < lefts[1, 0, 1]:
        new_cnt[0, 0, 0] = lefts[0, 0, 0]
        new_cnt[0, 0, 1] = lefts[0, 0, 1]
        new_cnt[3, 0, 0] = lefts[1, 0, 0]
        new_cnt[3, 0, 1] = lefts[1, 0, 1]
    else:
        new_cnt[0, 0, 0] = lefts[1, 0, 0]
        new_cnt[0, 0, 1] = lefts[1, 0, 1]
        new_cnt[3, 0, 0] = lefts[0, 0, 0]
        new_cnt[3, 0, 1] = lefts[0, 0, 1]

    if rights[0, 0, 1] < rights[1, 0, 1]:
        new_cnt[1, 0, 0] = rights[0, 0, 0]
        new_cnt[1, 0, 1] = rights[0, 0, 1]
        new_cnt[2, 0, 0] = rights[1, 0, 0]
        new_cnt[2, 0, 1] = rights[1, 0, 1]
    else:
        new_cnt[1, 0, 0] = rights[1, 0, 0]
        new_cnt[1, 0, 1] = rights[1, 0, 1]
        new_cnt[2, 0, 0] = rights[0, 0, 0]
        new_cnt[2, 0, 1] = rights[0, 0, 1]
    return new_cnt


""" Be careful while modifying HSV filter range, you may 
    need to save warped images to the training set again """
def preprocessing_exchange(frame):
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
    # boolImg = (np.logical_and(np.logical_and(np.logical_or(hsvImg[:,:,0] <= 10, hsvImg[:,:,0] >= 150), hsvImg[:,:,1] >= 100), hsvImg[:,:,2] >= 100) * 255).astype(np.uint8)
    return boolImg, hsvImg


""" Be careful while modifying HSV filter range, you may 
    need to save warped images to the training set again """
def preprocessing(frame):
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
model.load_state_dict(torch.load(file_path + '/simple_digits_classification/model.pth'))
model.eval()
def classify(image):
    # `image`: grayscale image
    image = cv2.resize(image, (50, 50))
    x = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
    logits = model(x)
    idx = torch.argmax(logits, dim=1).item()
    return idx

def square_detection(grayImg, camera_matrix, area_filter_size=30, height_range=(-10000.0, 200000.0)):
    projection_points = True
    quads = []
    quads_f = []

    contours, hierarchy = cv2.findContours(
        grayImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    testImg:np.ndarray = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=False)
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h/w > 1.5 or w/h > 1.5:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # print(len(approx))
            if len(approx) != 4:
                continue
            """ find warped rect """
            frame = cv2.drawContours(testImg.copy(), [approx], -1, (0, 255, 0), 1)
            # """ for debug """
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)

            quads.append(approx)
            quads_f.append(approx.astype(float))

            src = approx.astype(np.float32)
            l = 200
            dst = np.array([[l-1, 0], [0, 0], [0, l-1], [l-1, l-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)  # 获取变换矩阵
            warped = cv2.warpPerspective(grayImg, M, (l, l))  # 进行变换
            warped = cv2.bitwise_not(warped)
            cv2.imshow(f"warped {i}", warped)
            if (cv2.waitKey(0) == ord('s')):
                cv2.imwrite(f"warped_{i}.png", warped)
            print(f"warped {i}, classified: {classify(warped)}")


    if projection_points:
        rvec_list = []
        tvec_list = []
        quads_prj = []
        area_list = []

        block_size = 0.045
        model_object = np.array(
            [
                (0 - 0.5 * block_size, 0 - 0.5 * block_size, 0.0),
                (block_size - 0.5 * block_size, 0 - 0.5 * block_size, 0.0),
                (block_size - 0.5 * block_size, block_size - 0.5 * block_size, 0.0),
                (0 - 0.5 * block_size, block_size - 0.5 * block_size, 0.0),
            ]
        )

        dist_coeffs = np.array([[0, 0, 0, 0]], dtype="double")
        for quad in quads_f:
            model_image = np.squeeze(quad)
            ret, rvec, tvec = cv2.solvePnP(
                model_object, model_image, camera_matrix, dist_coeffs
            )
            projectedPoints, _ = cv2.projectPoints(
                model_object, rvec, tvec, camera_matrix, dist_coeffs
            )

            err = 0
            for t in range(len(projectedPoints)):
                err += np.linalg.norm(projectedPoints[t] - model_image[t])

            area = cv2.contourArea(quad.astype(np.int))
            if (
                err / area < 0.005
                and tvec[1] > height_range[0]
                and tvec[1] < height_range[1]
            ):
                quads_prj.append(projectedPoints.astype(int))
                rvec_list.append(rvec)
                tvec_list.append(tvec)
                area_list.append(area)
        return quads_prj, tvec_list, rvec_list, area_list, quads
    else:
        return (
            quads,
            [[0, 0, 0] for _ in quads],
            [[0, 0, 0] for _ in quads],
            [cv2.contourArea(quad.astype(np.int)) for quad in quads],
            quads,
        )


def classification(frame, quads, template_ids=range(1, 9)):
    quads_ID = []
    minpoints_list = []
    wrapped_img_list = []
    for i in range(len(quads)):
        points_src = np.array(
            [
                [(quads[i][0, 0, 0], quads[i][0, 0, 1])],
                [(quads[i][1, 0, 0], quads[i][1, 0, 1])],
                [(quads[i][2, 0, 0], quads[i][2, 0, 1])],
                [(quads[i][3, 0, 0], quads[i][3, 0, 1])],
            ],
            dtype="float32",
        )

        points_dst = np.array([[0, 0], [49, 0], [49, 49], [0, 49]], dtype="float32")
        out_img = cv2.warpPerspective(
            frame, cv2.getPerspectiveTransform(points_src, points_dst), (50, 50)
        )
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        out_img = cv2.threshold(out_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        wrapped_img_list.append(out_img)

        resize = False
        if resize:
            try:
                out_img[:3, :] = 0
                out_img[47:, :] = 0
                out_img[:, :3] = 0
                out_img[:, 47:] = 0
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    out_img
                )
                for label_i in range(1, num_labels):
                    if stats[label_i, cv2.CC_STAT_AREA].astype(float) < 35:  # 原50
                        out_img[labels == label_i] = 0

                nonzero_img = np.nonzero(out_img)
                left, right = np.min(nonzero_img[0]), np.max(nonzero_img[0])
                top, bottom = np.min(nonzero_img[1]), np.max(nonzero_img[1])
                right, bottom = min(right + 1, 49), min(bottom + 1, 49)
                nonzero_img = out_img[left:right, top:bottom]
                nonzero_img = cv2.resize(
                    nonzero_img, (36, 36), interpolation=cv2.INTER_NEAREST
                )
                out_img = np.zeros((50, 50), dtype=np.uint8)
                out_img[7 : 7 + 36, 7 : 7 + 36] = nonzero_img
            except:
                rospy.loginfo("resize trick failed, back to original img as tempate")
        out_img = map_img77(out_img)

        match_candidate = []
        match_candidate.append(out_img)
        match_candidate.append(cv2.rotate(out_img, cv2.ROTATE_180))
        match_candidate.append(cv2.rotate(out_img, cv2.ROTATE_90_CLOCKWISE))
        match_candidate.append(cv2.rotate(out_img, cv2.ROTATE_90_COUNTERCLOCKWISE))

        min_diff = 10000
        min_diff_target = 0

        for tid in template_ids:
            for tt in range(4):
                diff_img = cv2.absdiff(templates[tid - 1], match_candidate[tt])
                sum = np.sum(diff_img) / 255.0 / diff_img.size
                if min_diff > sum:
                    min_diff = sum
                    min_diff_target = tid

        if min_diff < 0.2:
            quads_ID.append(min_diff_target)
            minpoints_list.append(min_diff)
        else:
            quads_ID.append(-1)
            minpoints_list.append(min_diff)

    return quads_ID, minpoints_list, wrapped_img_list


def marker_detection(
    frame,
    camera_matrix,
    template_ids=range(1, 9),
    area_filter_size=30,
    seg_papram=None,
    verbose=True,
    height_range=(-10000.0, 200000.0),
    exchange_station=False,
):
    if exchange_station:
        tframe = copy.deepcopy(frame)
        tframe[int(tframe.shape[0] * 0.32) :, :, :] = 0
        boolImg, _ = preprocessing_exchange(tframe)
    else:
        boolImg, _ = preprocessing(frame)

    # cv2.imshow("boolImg", copy.deepcopy(boolImg))
    # cv2.waitKey(3)
    
    quads, tvec_list, rvec_list, area_list, ori_quads = square_detection(
        boolImg, camera_matrix, area_filter_size=area_filter_size, height_range=height_range
    )
    quads_ID, minpoints_list, wrapped_img_list = classification(
        frame, quads, template_ids=template_ids
    )
    if verbose:
        id = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "B", 7: "O", 8: "X", -1: "*"}
        for i in range(len(quads)):
            bbox = cv2.boundingRect(quads[i])
            try:
                cv2.putText(
                    frame,
                    id[quads_ID[i]],
                    (bbox[0], bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            except:
                traceback.print_exc()
        cv2.drawContours(frame, quads, -1, (0, 255, 0), 1)
    ids = [i for i in range(len(quads_ID)) if quads_ID[i] >= 1 and quads_ID[i] <= 8]
    return (
        [quads_ID[_] for _ in ids],
        [quads[_] for _ in ids],
        [area_list[_] for _ in ids],
        [tvec_list[_] for _ in ids],
        [rvec_list[_] for _ in ids],
        wrapped_img_list,
        minpoints_list,
    )
