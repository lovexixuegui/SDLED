import os
import cv2
import numpy as np
import time
from utils import line_ske


def accuracy(image_path,label_contours_path,save_path):
    res_path = os.path.join(save_path, "res")
    ske_path = os.path.join(save_path, "ske")
    ske_temp_path = os.path.join(save_path, "ske_temp")
    ske_png_path = os.path.join(save_path, "ske_png")
    complete_ske_png_path = os.path.join(save_path, "complete_ske_png")
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
    if not os.path.exists(ske_png_path):
        os.mkdir(ske_png_path)
    if not os.path.exists(ske_result_path):
        os.mkdir(ske_result_path)
    if not os.path.exists(complete_ske_png_path):
        os.mkdir(complete_ske_png_path)
    if not os.path.exists(complete_ske_result_path):
        os.mkdir(complete_ske_result_path)

    # Skeleton edge strength map
    # skeleton.skeleton2(res_path,ske_path,ske_temp_path)
    line_ske.ske(res_path)

    files = os.listdir(res_path)
    for file in files:
        res_name = os.path.join(res_path, file)
        image_name = os.path.join(image_path, file)
        label_contours_name = os.path.join(label_contours_path, png_file)
        ske_name = os.path.join(ske_path, file)
        ske_result_name = os.path.join(ske_result_path, png_file)
        complete_ske_png_name = os.path.join(complete_ske_png_path, png_file)
        complete_ske_result_name = os.path.join(complete_ske_result_path, file)

        image = cv2.imread(image_name, 1)
        ske = cv2.imread(ske_name, 0)
        image[:, :, 0] = np.where(ske == 1, 0, image[:, :, 0])
        image[:, :, 1] = np.where(ske == 1, 0, image[:, :, 1])
        image[:, :, 2] = np.where(ske == 1, 255, image[:, :, 2])
        cv2.imwrite(ske_result_name, image)

        ske_png_name = os.path.join(ske_png_path, png_file)
        ske_img = cv2.imread(ske_png_name, 0)
        # print(ske_png_name)
        singlePoints2 = myWay.findSinglePoints(ske_img, symbol=255)
        needRepair, index = myWay.findNeedRepair(ske_img, singlePoints2, symbol=255)
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

                        distance = myWay.pointsDistance(x_np[i], y_np[i], x_np[m], y_np[m])
                        # print((x_np[i], y_np[i], x_np[m], y_np[m]))#(335, 196, 343, 196)
                        # print("distance:" + str(distance))
                        if myWay.isEdgePoints(ske_img, x_np[i], y_np[i], x_np[m], y_np[m], edge_limit=5):
                            ske_img = myWay.repair_limit_edge2(ske_img, x_np[i], y_np[i], x_np[m], y_np[m],
                                                               edge_limit=5, symbol=255)
                index -= 1
                number = 0
                x_np[number] = x
                y_np[number] = y
                number += 1

        # for i in range(len(ske_img)):
        #     ske_img[i][0] = 255
        #     ske_img[i][len(ske_img[0]) - 1] = 255
        # for m in range(len(ske_img[0])):
        #     ske_img[0][m] = 255
        #     ske_img[len(ske_img) - 1][m] = 255

        singlePoints = myWay.findSinglePoints(ske_img)
        while (len(singlePoints) > 0):
            ske_img = myWay.deleteSingleWay(ske_img, singlePoints, final=True)
            singlePoints = myWay.findSinglePoints(ske_img)
        cv2.imwrite(complete_ske_png_name, ske_img)

        image = cv2.imread(image_name, 1)
        image[:, :, 0] = np.where(ske_img == 1, 0, image[:, :, 0])
        image[:, :, 1] = np.where(ske_img == 1, 0, image[:, :, 1])
        image[:, :, 2] = np.where(ske_img == 1, 255, image[:, :, 2])
        cv2.imwrite(complete_ske_result_name, image)

    functions.line_IoU(ske_png_path, label_contours_path, line_iou_kernel5_txt_path, kernel=5, threshold=255)
    functions.line_IoU(ske_png_path, label_contours_path, line_iou_kernel3_txt_path, kernel=3, threshold=255)
    functions.line_IoU(complete_ske_png_path, label_contours_path, complete_line_iou_kernel5_txt_path, kernel=5,
                       threshold=255)
    functions.line_IoU(complete_ske_png_path, label_contours_path, complete_line_iou_kernel3_txt_path, kernel=3,
                       threshold=255)


# skeleton.skeleton2(res_path,ske_path,ske_temp_path)
time.sleep(1)
ske_files=os.listdir(ske_path)
for ske_file in ske_files:
    ske_old_name=os.path.join(ske_path,ske_file)
    ske_new_name=os.path.join(ske_png_path,ske_file)
    # print(ske_old_name)
    ske_new_name=ske_new_name.replace(".tif",".png")
    # print(ske_new_name)
    ske_old_img=cv2.imread(ske_old_name,0)
    ske_new_img=np.where(ske_old_img==0,0,255)
    cv2.imwrite(ske_new_name,ske_new_img)

res_files=os.listdir(res_path)
for res_file in res_files:
    s_name=os.path.join(ske_png_path,res_file)
    s_name=s_name.replace(".tif",".png")
    t_name=os.path.join(ske_path,res_file)
    t_png_name=t_name.replace(".tif",".png")
    if not os.path.exists(s_name):
        n_img=np.zeros((512,512,3))
        # print(s_name)
        cv2.imwrite(s_name,n_img)
        t_img=np.zeros((512,512))
        cv2.imwrite(t_png_name,t_img)
        os.rename(t_png_name,t_name)

files=os.listdir(res_path)
for file in files:
    png_file=file.replace(".tif",".png")
    res_name=os.path.join(res_path,file)
    image_name=os.path.join(image_path,file)
    label_contours_name=os.path.join(label_contours_path,png_file)
    ske_name=os.path.join(ske_path,file)
    ske_result_name=os.path.join(ske_result_path,png_file)
    complete_ske_png_name=os.path.join(complete_ske_png_path,png_file)
    complete_ske_result_name=os.path.join(complete_ske_result_path,file)

    image = cv2.imread(image_name, 1)
    ske = cv2.imread(ske_name, 0)
    image[:, :, 0] = np.where(ske == 1, 0, image[:, :, 0])
    image[:, :, 1] = np.where(ske == 1, 0, image[:, :, 1])
    image[:, :, 2] = np.where(ske == 1, 255, image[:, :, 2])
    cv2.imwrite(ske_result_name, image)





    ske_png_name=os.path.join(ske_png_path,png_file)
    ske_img = cv2.imread(ske_png_name, 0)
    # print(ske_png_name)
    singlePoints2 = myWay.findSinglePoints(ske_img, symbol=255)
    needRepair,index = myWay.findNeedRepair(ske_img, singlePoints2, symbol=255)
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

                    distance = myWay.pointsDistance(x_np[i], y_np[i], x_np[m], y_np[m])
                    # print((x_np[i], y_np[i], x_np[m], y_np[m]))#(335, 196, 343, 196)
                    # print("distance:" + str(distance))
                    if myWay.isEdgePoints(ske_img, x_np[i], y_np[i], x_np[m], y_np[m],edge_limit=5):
                        ske_img=myWay.repair_limit_edge2(ske_img, x_np[i], y_np[i], x_np[m], y_np[m],edge_limit=5,symbol=255)
            index -= 1
            number = 0
            x_np[number] = x
            y_np[number] = y
            number += 1


    # for i in range(len(ske_img)):
    #     ske_img[i][0] = 255
    #     ske_img[i][len(ske_img[0]) - 1] = 255
    # for m in range(len(ske_img[0])):
    #     ske_img[0][m] = 255
    #     ske_img[len(ske_img) - 1][m] = 255

    singlePoints = myWay.findSinglePoints(ske_img)
    while (len(singlePoints) > 0):
        ske_img = myWay.deleteSingleWay(ske_img, singlePoints, final=True)
        singlePoints = myWay.findSinglePoints(ske_img)
    cv2.imwrite(complete_ske_png_name,ske_img)

    image = cv2.imread(image_name, 1)
    image[:, :, 0] = np.where(ske_img == 1, 0, image[:, :, 0])
    image[:, :, 1] = np.where(ske_img == 1, 0, image[:, :, 1])
    image[:, :, 2] = np.where(ske_img == 1, 255, image[:, :, 2])
    cv2.imwrite(complete_ske_result_name, image)

functions.line_IoU(ske_png_path,label_contours_path,line_iou_kernel5_txt_path,kernel=5,threshold=255)
functions.line_IoU(ske_png_path,label_contours_path,line_iou_kernel3_txt_path,kernel=3,threshold=255)
functions.line_IoU(complete_ske_png_path,label_contours_path,complete_line_iou_kernel5_txt_path,kernel=5,threshold=255)
functions.line_IoU(complete_ske_png_path,label_contours_path,complete_line_iou_kernel3_txt_path,kernel=3,threshold=255)

#end