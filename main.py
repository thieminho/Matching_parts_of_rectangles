#!/usr/bin/python
import os
import sys
import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr
import math
import re


def check_if_int(s):
    try:
        return int(s)
    except:
        return s


def split_to_numbers_and_words(s):
    return [check_if_int(c) for c in re.split('([0-9]+)', s)]


def rotate(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_canny = cv2.Canny(gray_scale, 100, 200)
    lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180.0, 60, minLineLength=10, maxLineGap=5)
    # print(lines)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        # print(angles)
        median_angle = np.median(angles)
        rotated = ndimage.rotate(image, median_angle)
        if rotated.shape[0] > rotated.shape[1]:
            rotated = ndimage.rotate(rotated, 90)
        return rotated
    else:
        return image


def cut(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)
    top_bottom = thresh[np.any(thresh, axis=1), :]
    # cv2.imshow('image', top_bottom)
    # cv2.waitKey(0)
    left_right = top_bottom[:, np.any(top_bottom, axis=0)]
    # cv2.imshow('image', left_right)
    # cv2.waitKey(0)
    return left_right


def normalize(image, kernel, size):
    after_blur = cv2.GaussianBlur(image, kernel, 0)
    resized = cv2.resize(after_blur, size, interpolation=cv2.INTER_LINEAR)
    if np.sum(resized[-3, :]) < np.sum(resized[3, :]):
        resized = cv2.flip(resized, -1)
    _, thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    normalized = thresh[np.any(thresh is False, axis=1), :]
    return normalized


def cut2(image):
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    cut_image = image[:450, 50:-50]
    # cv2.imshow('image', cut_image)
    # cv2.waitKey(0)
    _, thresh = cv2.threshold(cut_image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)
    norm = cv2.resize(thresh, (735, thresh.shape[0]), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('image', norm)
    # cv2.waitKey(0)
    return norm


def description_of_image(image, step, sensitivity):
    description = []
    for i in range(image.shape[1] // step):
        column = image[:, i * step]
        sum_of_pixels_in_one_column = (column > sensitivity).sum()
        description.append(sum_of_pixels_in_one_column)
    # print(description)
    return description


def compare(descriptions):
    distances = []
    descriptions_np = np.asarray(descriptions)

    for i in range(len(descriptions)):
        distance_from_one = []
        row = descriptions_np[i, :]
        for k in range(descriptions_np.shape[0]):
            reversed = descriptions_np[k, ::-1]
            correlation = pearsonr(row, reversed)
            if i == k:
                distance_from_one.append((9999, 9999))
            else:
                distance_from_one.append(correlation)
        distances.append(distance_from_one)
    return distances


if __name__ == "__main__":
    # path = "./set4"
    # number = 20
    path = sys.argv[1]
    number = int(sys.argv[2])
    images = []
    files = sorted(os.listdir(path), key=split_to_numbers_and_words)
    for i in range(number):
        image = cv2.imread(os.path.join(path, files[i]))
        if image is not None:
            images.append(image)
    descriptions = []

    for image in images:
        rotated = rotate(image)
        rotated_cut = cut(rotated)
        dimension = int(rotated_cut.shape[0] / 25)
        if dimension > 2 and dimension % 2 == 0:
            dimension -= 1
        elif dimension <= 2:
            dimension = 3

        normalized = normalize(rotated_cut, (dimension, dimension), (735, 500))
        # cv2.imshow('image', normalized)
        # cv2.waitKey(0)
        image = cut2(normalized)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        des = description_of_image(image, 3, 200)
        descriptions.append(des)

    desc_t = np.asarray(descriptions)
    distances = compare(descriptions)

    for i, dist in enumerate(distances):
        matching = sorted(range(len(dist)), key=lambda k: dist[k])
        matching.remove(i)
        print(*matching, sep=' ')
