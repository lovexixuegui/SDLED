# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import morphology
import os

def ske(poly_path,threshold=128):
    binary_save_path = poly_path + str(threshold)
    ske_save_path = poly_path.replace("res", "ske")
    if not os.path.exists(binary_save_path):
        os.makedirs(binary_save_path)
    if not os.path.exists(ske_save_path):
        os.makedirs(ske_save_path)

    files = os.listdir(poly_path)
    for file in files:
        poly_name = os.path.join(poly_path, file)
        binary_save_name = os.path.join(binary_save_path, file)
        ske_save_name=os.path.join(ske_save_path,file)

        # binary edge strength map
        poly = cv2.imread(poly_name, cv2.IMREAD_GRAYSCALE)
        ret, poly = cv2.threshold(poly, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite(binary_save_name, poly)
        cv2.waitKey(0)


        ret, thresh = cv2.threshold(poly, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        skeleton = morphology.skeletonize(thresh)
        ske = np.multiply(thresh, skeleton)
        ret, thresh = cv2.threshold(ske, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(ske_save_name, thresh)
        cv2.waitKey(0)


if __name__ == "__main__":
    ske(r"F:\zhang_weak\test\res",threshold=100)