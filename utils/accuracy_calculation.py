import os
import cv2
import numpy as np
import time
from utils import line_ske
from utils import post_process
from utils import functions
from utils import skeleton

def accuracy(image_path,label_contours_path,poly_label_path,save_path):
    res_path = os.path.join(save_path, "res")
    ske_path = os.path.join(save_path, "ske")
    ske_temp_path = os.path.join(save_path, "ske_temp")
    # ske_png_path = os.path.join(save_path, "ske_png")
    complete_ske_path = os.path.join(save_path, "complete_ske")
    complete_ske_result_path = os.path.join(save_path, "complete_ske_result")
    ske_result_path = os.path.join(save_path, "ske_result")
    line_iou_kernel5_txt_path = os.path.join(save_path, "line_iou_kernel5.txt")
    line_iou_kernel3_txt_path = os.path.join(save_path, "line_iou_kernel3.txt")
    complete_line_iou_kernel5_txt_path = os.path.join(save_path, "complete_line_iou_kernel5.txt")
    complete_line_iou_kernel3_txt_path = os.path.join(save_path, "complete_line_iou_kernel3.txt")

    if not os.path.exists(ske_path):
        os.mkdir(ske_path)
    if not os.path.exists(ske_temp_path):
        os.mkdir(ske_temp_path)
    if not os.path.exists(ske_result_path):
        os.mkdir(ske_result_path)
    if not os.path.exists(complete_ske_path):
        os.mkdir(complete_ske_path)
    if not os.path.exists(complete_ske_result_path):
        os.mkdir(complete_ske_result_path)

    # Skeleton edge strength map
    skeleton.skeleton2(res_path,ske_path,ske_temp_path)
    # line_ske.ske(res_path)

    files = os.listdir(res_path)
    for file in files:
        res_name = os.path.join(res_path, file)
        image_name = os.path.join(image_path, file)
        label_contours_name = os.path.join(label_contours_path, file)
        ske_name = os.path.join(ske_path, file)
        ske_result_name = os.path.join(ske_result_path, file)
        complete_ske_name = os.path.join(complete_ske_path, file)
        complete_ske_result_name = os.path.join(complete_ske_result_path, file)

        image = cv2.imread(image_name, 1)
        ske = cv2.imread(ske_name, 0)
        image[:, :, 0] = np.where(ske == 0, image[:, :, 0], 0)
        image[:, :, 1] = np.where(ske == 0, image[:, :, 1], 0)
        image[:, :, 2] = np.where(ske == 0, image[:, :, 2], 255)
        cv2.imwrite(ske_result_name, image)


        # make the building edge complete, which near the image edge
        singlePoints2 = post_process.findSinglePoints(ske, symbol=255)
        needRepair, index = post_process.findNeedRepair(ske, singlePoints2, symbol=255)
        x_np = np.zeros((2000), int)
        y_np = np.zeros((2000), int)
        number = 0
        len_repair = len(needRepair)
        for n in range(len_repair):
            x, y, temp_index = needRepair.pop()
            if index == temp_index:
                x_np[number] = x
                y_np[number] = y
                number += 1
            if index != temp_index or n == len_repair - 1:
                for i in range(number - 1):
                    for m in range(i + 1, number):
                        if post_process.isEdgePoints(ske, x_np[i], y_np[i], x_np[m], y_np[m], edge_limit=5):
                            ske_img = post_process.repair_limit_edge2(ske, x_np[i], y_np[i], x_np[m], y_np[m],
                                                               edge_limit=5, symbol=255)
                index -= 1
                number = 0
                x_np[number] = x
                y_np[number] = y
                number += 1

        # delete the incomplete edge
        singlePoints = post_process.findSinglePoints(ske_img)
        while (len(singlePoints) > 0):
            ske_img = post_process.deleteSingleWay(ske_img, singlePoints, final=True)
            singlePoints = post_process.findSinglePoints(ske_img)
        cv2.imwrite(complete_ske_name, ske_img)

        image = cv2.imread(image_name, 1)
        image[:, :, 0] = np.where(ske_img == 0, image[:, :, 0], 0)
        image[:, :, 1] = np.where(ske_img == 0, image[:, :, 1], 0)
        image[:, :, 2] = np.where(ske_img == 0, image[:, :, 2], 255)
        cv2.imwrite(complete_ske_result_name, image)

    functions.F1_score(image_path,label_path,poly_label_path,save_path)
    functions.poly_IoU_inBBox(image_path,label_path,poly_label_path,save_path)
    functions.all_poly_IoU(image_path,label_path,poly_label_path,save_path)
    functions.line_IoU_inBBox(image_path,label_path,poly_label_path,save_path,kernel=3)
    functions.line_IoU_inBBox(image_path,label_path,poly_label_path,save_path,kernel=5)

    # functions.line_IoU(ske_path, label_contours_path, line_iou_kernel5_txt_path, kernel=5, threshold=255)
    # functions.line_IoU(ske_path, label_contours_path, line_iou_kernel3_txt_path, kernel=3, threshold=255)
    # functions.line_IoU(complete_ske_path, label_contours_path, complete_line_iou_kernel5_txt_path, kernel=5,
    #                    threshold=255)
    # functions.line_IoU(complete_ske_path, label_contours_path, complete_line_iou_kernel3_txt_path, kernel=3,
    #                    threshold=255)


if __name__ == "__main__":
    # networkx==2.3 (networkx==2.5 may have some problem)
    image_path=r"F:\zhang_weak\test3\image"
    label_path=r"F:\zhang_weak\test3\label"
    poly_label_path=r"F:\zhang_weak\test3\poly_label"
    save_path=r"F:\zhang_weak\test3"
    accuracy(image_path,label_path,poly_label_path,save_path)