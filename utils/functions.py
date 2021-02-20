import os
import copy
import cv2
import numpy as np
import gdal
import shutil

def delete_inline_rect_inTXT(old_txt_path,new_txt_path):
    files = os.listdir(old_txt_path)
    for file in files:
        # print(file)
        old_txt_name = os.path.join(old_txt_path, file)
        new_txt_name = os.path.join(new_txt_path, file)
        f1 = open(old_txt_name, "r")
        f2 = open(new_txt_name, "w")
        coordinate = []
        for line in f1:
            y1 = int(line.split(" ")[0])
            x1 = int(line.split(" ")[1])
            y2 = int(line.split(" ")[2])
            x2 = line.split(" ")[3]
            x2 = int(x2.split("/n")[0])
            if (len(coordinate) > 0):
                temp_coordinate = copy.deepcopy(coordinate)
                panduan = 0
                while (len(temp_coordinate) > 0):
                    ty1, tx1, ty2, tx2 = temp_coordinate.pop()
                    if (ty1 <= y1 and tx1 <= x1 and ty2 >= y2 and tx2 >= x2):
                        panduan = 1
                        break
                if (panduan == 0):
                    f2.write(str(y1) + " " + str(x1) + " " + str(y2) + " " + str(x2) + "\n")
                    coordinate.append((y1, x1, y2, x2))
            else:
                f2.write(str(y1) + " " + str(x1) + " " + str(y2) + " " + str(x2) + "\n")
                coordinate.append((y1, x1, y2, x2))

def fill_poly_without_inline_part(result_path,save_path):
    files = os.listdir(result_path)
    for file in files:
        # print(file)
        result_name = os.path.join(result_path, file)
        save_name = os.path.join(save_path, file)
        img = cv2.imread(result_name, 0)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp_num = 0
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:
                cv2.fillPoly(img, [contours[i]], 255)
                if (i + 1 < len(contours) and hierarchy[0][i - 1][3] != -1):
                    if (hierarchy[0][i + 1][3] != hierarchy[0][i][3] and temp_num == 1):
                        cv2.fillPoly(img, [contours[i]], 0)
                        temp_num = 0
                    elif (hierarchy[0][i + 1][3] != hierarchy[0][i][3] and temp_num == 0):
                        temp_num = 1
            else:
                temp_num = 0
        cv2.imwrite(save_name, img)

def poly_IoU(result_path,label_edge_path,txt_path,poly_iou_txt_path):
    all_sum_IoU = 0
    iou_txt = open(poly_iou_txt_path, "w")
    iou_txt.write(result_path + "\n")
    files = os.listdir(result_path)
    for file in files:
        result_name = os.path.join(result_path, file)
        label_name = os.path.join(label_edge_path, file)
        txt_file = file.replace(".png", ".txt")
        rect_txt_name=os.path.join(txt_path,txt_file)
        result_fillPoly = cv2.imread(result_name, 0)
        label = cv2.imread(label_name, 0)
        rect_txt = open(rect_txt_name, "r")

        sum_IoU = 0
        number = 0
        sumIsZero = 1
        for line in rect_txt:
            number += 1
            y1 = int(line.split(" ")[0])
            x1 = int(line.split(" ")[1])
            y2 = int(line.split(" ")[2])
            x2 = line.split(" ")[3]
            x2 = int(x2.split("/n")[0])
            overLap = np.where((result_fillPoly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0) &
                               (label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0))[0].size
            u1 = np.where((result_fillPoly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0) &
                          (label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0))[0].size
            u2 = np.where((result_fillPoly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0) &
                          (label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0))[0].size
            sum = overLap + u1 + u2
            if sum != 0:
                IoU = overLap / sum
            else:
                IoU = 0
            sum_IoU += IoU
            if sum > 0: sumIsZero = 0

        if number != 0:
            ave_IoU = sum_IoU / number
        else:
            if sumIsZero == 0:
                ave_IoU = 0
            else:
                ave_IoU = 1
        iou_txt.write(file + " :" + str(ave_IoU) + "\n")
        all_sum_IoU += ave_IoU
        # break
    iou_txt.close()
    # iou_txt.write("Ave_IoU :" + str(all_sum_IoU / len(files)) + "\n")
    with open(poly_iou_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_IoU :" + str(all_sum_IoU / len(files)) + "\n" + content)
    f.close()

def line_IoU(result_path,label_edge_path,iou_txt_path,kernel=5,threshold=255):
    sum_IoU = 0
    iou_txt = open(iou_txt_path, "w")
    iou_txt.write(result_path + "\n")
    files = os.listdir(result_path)
    for file in files:
        result_name = os.path.join(result_path, file)
        label_name = os.path.join(label_edge_path, file)
        result_img = cv2.imread(result_name, 0)
        label_img = cv2.imread(label_name, 0)
        dilate_kernel = np.ones((kernel, kernel), np.uint8)
        result_dilation = cv2.dilate(result_img, dilate_kernel)
        label_dilation = cv2.dilate(label_img, dilate_kernel)
        overLap = np.where((result_dilation[:, :] == threshold) & (label_dilation[:, :] == threshold))[0].size
        u1 = np.where((result_dilation[:, :] == 0) & (label_dilation[:, :] == threshold))[0].size
        u2 = np.where((result_dilation[:, :] == threshold) & (label_dilation[:, :] == 0))[0].size
        sum = overLap + u1 + u2
        if sum != 0:
            IoU = overLap / sum
        else:
            IoU = 0
        sum_IoU += IoU
        iou_txt.write(file + " :" + str(IoU) + "\n")
    iou_txt.close()
    with open(iou_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_IoU :" + str(round(sum_IoU / len(files),4)) + "\n" + content)
    f.close()


def drawContours(label_path,save_path):
    files = os.listdir(label_path)
    for file in files:
        mask_prediction_name = os.path.join(label_path, file)
        save_name = os.path.join(save_path, file)
        img = cv2.imread(mask_prediction_name, 0)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        save_img = np.zeros((img.shape[0], img.shape[1]))
        cv2.drawContours(save_img, contours, -1, (255, 255, 255), 1)
        cv2.imwrite(save_name, save_img)

def label_tif_To_png(tif_path,save_path):
    files = os.listdir(tif_path)
    for file in files:
        tif_name = os.path.join(tif_path, file)
        save_name = os.path.join(save_path, file)
        save_name = save_name.replace(".tif", ".png")
        if(file.find(".png")!=-1):
            shutil.copy(tif_name,save_name)
        else:
            dataset = gdal.Open(tif_name)
            if dataset == None:
                print(tif_name + "文件无法打开")
            im_width = dataset.RasterXSize  # 栅格矩阵的列数
            im_height = dataset.RasterYSize  # 栅格矩阵的行数
            im_bands = dataset.RasterCount  # 波段数
            im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
            save_img = np.where(im_data >0, 255, 0)
            cv2.imwrite(save_name, save_img)

def poly_IoU_with_draw_result(img_path,result_path,result_fillPoly_path,label_path,rect_txt_path,save_path,iou_txt_path):
    all_sum_IoU = 0
    iou_txt = open(iou_txt_path, "w")
    iou_txt.write(result_path + "\n")
    files = os.listdir(result_path)
    for file in files:
        file = file.replace(".tif", ".png")
        tif_file = file.replace(".png", ".tif")
        print(file)
        txt_file = file.replace(".png", ".txt")
        img_name = os.path.join(img_path, file)
        # #dinknet
        # result_name=os.path.join(result_path,tif_file)
        # result_fillPoly_name=os.path.join(result_fillPoly_path,tif_file)

        # maskrcnn
        result_name = os.path.join(result_path, file)
        result_fillPoly_name = os.path.join(result_fillPoly_path, file)

        label_name = os.path.join(label_path, file)
        rect_txt_name = os.path.join(rect_txt_path, txt_file)
        save_name = os.path.join(save_path, file)

        rect_txt = open(rect_txt_name, "r")
        img = cv2.imread(img_name)
        result = cv2.imread(result_name, 0)
        result_fillPoly = cv2.imread(result_fillPoly_name, 0)
        label = cv2.imread(label_name, 0)
        sum_IoU = 0
        number = 0
        sumIsZero = 1
        for line in rect_txt:
            # print(line)
            number += 1
            y1 = int(line.split(" ")[0])
            x1 = int(line.split(" ")[1])
            y2 = int(line.split(" ")[2])
            x2 = line.split(" ")[3]
            x2 = int(x2.split("/n")[0])
            # x1 = int(line.split(" ")[0])
            # y1 = int(line.split(" ")[1])
            # x2 = int(line.split(" ")[2])
            # y2 = line.split(" ")[3]
            # y2 = int(y2.split("/n")[0])

            # 计算IoU精度，注意X和Y的顺序
            overLap = np.where((result_fillPoly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0) &
                               (label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0))[0].size
            u1 = np.where((result_fillPoly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0) &
                          (label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0))[0].size
            u2 = np.where((result_fillPoly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0) &
                          (label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0))[0].size
            sum = overLap + u1 + u2
            if sum != 0:
                IoU = overLap / sum
            else:
                IoU = 0
            # print("IoU:" +str(IoU))
            sum_IoU += IoU
            if sum > 0: sumIsZero = 0

            # 画外接矩形框
            if x2 == img.shape[0]: x2 = img.shape[0] - 1
            if y2 == img.shape[1]: y2 = img.shape[1] - 1
            img[x1:x2 + 1, (y1, y2), 0] = 0
            img[x1:x2 + 1, (y1, y2), 1] = 255
            img[x1:x2 + 1, (y1, y2), 2] = 0
            img[(x1, x2), y1:y2 + 1, 0] = 0
            img[(x1, x2), y1:y2 + 1, 1] = 255
            img[(x1, x2), y1:y2 + 1, 2] = 0

            # 画实验结果边界
            img[:, :, 0] = np.where(result == 255, 0, img[:, :, 0])
            img[:, :, 1] = np.where(result == 255, 0, img[:, :, 1])
            img[:, :, 2] = np.where(result == 255, 255, img[:, :, 2])

            # 在矩形框内输入精度
            cv2.putText(img, '%.4f' % IoU, (y1, x1), fontFace=1, fontScale=1, color=(255, 0, 0), lineType=1)
            # cv2.imwrite(save_name, img)
            # break

        cv2.imwrite(save_name, img)
        if number != 0:
            ave_IoU = sum_IoU / number
        else:
            if sumIsZero == 0:
                ave_IoU = 0
            else:
                ave_IoU = 1
        iou_txt.write(file + " :" + str(ave_IoU) + "\n")
        all_sum_IoU += ave_IoU
        # break
    iou_txt.write("Ave_IoU :" + str(all_sum_IoU / len(files)) + "\n")

def bounding_rect_txt(label_path,txt_path):
    files = os.listdir(label_path)
    for file in files:
        label_name = os.path.join(label_path, file)
        # save_name = os.path.join(save_path, file)
        txt_name = os.path.join(txt_path, file)
        txt_name = txt_name.replace(".png", ".txt")
        txt = open(txt_name, "w")
        img = cv2.imread(label_name, 0)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            # cv2.rectangle(img, (x, y), (x + w, y + h), 255)
            txt.write(str(x) + " " + str(y) + " " + str(x + w) + " " + str(y + h) + "\n")

        # cv2.imwrite(save_name, img)
        txt.close()


def poly_IoU_inBBox(image_path,label_path,poly_label_path,save_path):
    complete_ske_path = os.path.join(save_path, "complete_ske")
    compelte_ske_poly_IoU_path=os.path.join(save_path,"complete_ske_polyIoU_inBBox")
    iou_txt_path = os.path.join(save_path, "complete_ske_polyIoU_inBBox.txt")
    if not os.path.exists(compelte_ske_poly_IoU_path):
        os.mkdir(compelte_ske_poly_IoU_path)

    all_sum_IoU = 0
    iou_txt = open(iou_txt_path, "w")
    iou_txt.write(complete_ske_path + "\n")
    files = os.listdir(image_path)
    for file in files:
        # txt_file = file.replace(".tif", ".txt")
        image_name = os.path.join(image_path, file)
        label_name = os.path.join(label_path, file)
        poly_label_name=os.path.join(poly_label_path,file)
        complete_ske_name = os.path.join(complete_ske_path, file)
        save_name = os.path.join(compelte_ske_poly_IoU_path, file)

        image=cv2.imread(image_name,1)
        complete_ske=cv2.imread(complete_ske_name,0)

        # fillPoly
        res_poly=cv2.imread(complete_ske_name,0)
        ske_contours, ske_hierarchy = cv2.findContours(res_poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp_num = 0
        for i in range(len(ske_contours)):
            if ske_hierarchy[0][i][3] != -1:
                cv2.fillPoly(res_poly, [ske_contours[i]], 255)
                if (i + 1 < len(ske_contours) and ske_hierarchy[0][i - 1][3] != -1):
                    if (ske_hierarchy[0][i + 1][3] != ske_hierarchy[0][i][3] and temp_num == 1):
                        cv2.fillPoly(res_poly, [ske_contours[i]], 0)
                        temp_num = 0
                    elif (ske_hierarchy[0][i + 1][3] != ske_hierarchy[0][i][3] and temp_num == 0):
                        temp_num = 1
            else:
                temp_num = 0

        label=cv2.imread(label_name,0)
        poly_label=cv2.imread(poly_label_name,0)
        print(image_name)
        contours, hierarchy = cv2.findContours(poly_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sum_IoU = 0
        number = 0
        sumIsZero = 1
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            number += 1
            y1 = x
            x1 = y
            y2 = x+w
            x2 = y+h

            # IoU
            overLap = np.where((res_poly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0) &
                               (poly_label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0))[0].size
            u1 = np.where((res_poly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0) &
                          (poly_label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0))[0].size
            u2 = np.where((res_poly[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0) &
                          (poly_label[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0))[0].size
            sum = overLap + u1 + u2
            if sum != 0:
                IoU = overLap / sum
            else:
                IoU = 0
            # print("IoU:" +str(IoU))
            sum_IoU += IoU
            if sum > 0: sumIsZero = 0

            # draw ground truth
            image[:, :, 0] = np.where(label == 0, image[:, :, 0], 0)
            image[:, :, 1] = np.where(label == 0, image[:, :, 1], 255)
            image[:, :, 2] = np.where(label == 0, image[:, :, 2], 0)

            # draw result
            image[:, :, 0] = np.where(complete_ske == 0, image[:, :, 0], 0)
            image[:, :, 1] = np.where(complete_ske == 0, image[:, :, 1], 0)
            image[:, :, 2] = np.where(complete_ske == 0, image[:, :, 2], 255)

            # # draw the overlap of ground truth and draw result
            # image[:, :, 0] = np.where((label == 0) & (complete_ske == 0), image[:, :, 0], 0)
            # image[:, :, 1] = np.where((label == 0) & (complete_ske == 0), image[:, :, 1], 255)
            # image[:, :, 2] = np.where((label == 0) & (complete_ske == 0), image[:, :, 2], 0)

            # tag IoU in bounding box
            cv2.putText(image, '%.4f' % IoU, (y1, x1), fontFace=1, fontScale=1, color=(255, 0, 0), lineType=1)

        cv2.imwrite(save_name, image)
        if number != 0:
            ave_IoU = sum_IoU / number
        else:
            if sumIsZero == 0:
                ave_IoU = 0
            else:
                ave_IoU = 1
        iou_txt.write(file + " :" + str(ave_IoU) + "\n")
        all_sum_IoU += ave_IoU

    iou_txt.close()
    with open(iou_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_IoU :" + str(round(all_sum_IoU / len(files), 4)) + "\n" + content)
    f.close()

def line_IoU_inBBox(image_path,label_path,poly_label_path,save_path,kernel=3):
    complete_ske_path = os.path.join(save_path, "complete_ske")
    iou_txt_path = os.path.join(save_path, "line_IoU_inBBox_kernel"+str(kernel)+".txt")

    all_sum_IoU = 0
    iou_txt = open(iou_txt_path, "w")
    iou_txt.write(complete_ske_path + "\n")
    files = os.listdir(image_path)
    for file in files:
        # image_name = os.path.join(image_path, file)
        label_name = os.path.join(label_path, file)
        poly_label_name=os.path.join(poly_label_path,file)
        complete_ske_name = os.path.join(complete_ske_path, file)
        # image=cv2.imread(image_name,1)
        complete_ske=cv2.imread(complete_ske_name,0)
        label=cv2.imread(label_name,0)
        poly_label=cv2.imread(poly_label_name,0)
        kernel_size = np.ones((kernel, kernel), np.uint8)
        result_dilation = cv2.dilate(complete_ske, kernel_size)
        label_dilation = cv2.dilate(label, kernel_size)

        contours, hierarchy = cv2.findContours(poly_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sum_IoU = 0
        number = 0
        sumIsZero = 1
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            number += 1
            y1 = x
            x1 = y
            y2 = x+w
            x2 = y+h

            overLap = np.where((result_dilation[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0.5) & (
                        label_dilation[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0.5))[0].size
            u1 = np.where((result_dilation[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0) & (
                        label_dilation[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0.5))[0].size
            u2 = np.where((result_dilation[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] > 0.5) & (
                        label_dilation[min(x1, x2):max(x1, x2) + 1, min(y1, y2):max(y1, y2) + 1] == 0))[0].size

            sum = overLap + u1 + u2
            if sum != 0:
                IoU = overLap / sum
            else:
                IoU = 0
            # print("IoU:" +str(IoU))
            sum_IoU += IoU
            if sum > 0: sumIsZero = 0

        if number != 0:
            ave_IoU = sum_IoU / number
        else:
            if sumIsZero == 0:
                ave_IoU = 0
            else:
                ave_IoU = 1
        iou_txt.write(file + " :" + str(ave_IoU) + "\n")
        all_sum_IoU += ave_IoU

    iou_txt.close()
    with open(iou_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_IoU :" + str(round(all_sum_IoU / len(files), 4)) + "\n" + content)
    f.close()

def F1_score(image_path,label_path,poly_label_path,save_path):
    complete_ske_path=os.path.join(save_path,"complete_ske")
    precision_txt_path=os.path.join(save_path,"precision.txt")
    recall_txt_path=os.path.join(save_path,"recall.txt")
    f1_txt_path=os.path.join(save_path,"F1_score.txt")
    precision_txt = open(precision_txt_path, "w")
    recall_txt = open(recall_txt_path, "w")
    f1_txt = open(f1_txt_path, "w")

    all_precision = 0
    all_recall = 0
    all_f1 = 0
    files = os.listdir(image_path)
    for file in files:
        complete_ske_name = os.path.join(complete_ske_path, file)
        poly_label_name=os.path.join(poly_label_path,file)

        # fillPoly
        res_poly=cv2.imread(complete_ske_name,0)
        ske_contours, ske_hierarchy = cv2.findContours(res_poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp_num = 0
        for i in range(len(ske_contours)):
            if ske_hierarchy[0][i][3] != -1:
                cv2.fillPoly(res_poly, [ske_contours[i]], 255)
                if (i + 1 < len(ske_contours) and ske_hierarchy[0][i - 1][3] != -1):
                    if (ske_hierarchy[0][i + 1][3] != ske_hierarchy[0][i][3] and temp_num == 1):
                        cv2.fillPoly(res_poly, [ske_contours[i]], 0)
                        temp_num = 0
                    elif (ske_hierarchy[0][i + 1][3] != ske_hierarchy[0][i][3] and temp_num == 0):
                        temp_num = 1
            else:
                temp_num = 0

        poly_label = cv2.imread(poly_label_name, 0)
        TP = np.where((poly_label > 0) & (res_poly > 0))[0].size
        FP = np.where((poly_label == 0) & (res_poly > 0))[0].size
        FN = np.where((poly_label > 0) & (res_poly == 0))[0].size

        if TP + FP == 0:
            precision = 1
        else:
            precision = TP / (TP + FP)
        if TP + FN == 0:
            recall = 1
        else:
            recall = TP / (TP + FN)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        precision_txt.write(file + " :" + str(precision) + "\n")
        recall_txt.write(file + " :" + str(recall) + "\n")
        f1_txt.write(file + " :" + str(f1) + "\n")
        all_precision += precision
        all_recall += recall
        all_f1 += f1

    precision_txt.close()
    recall_txt.close()
    f1_txt.close()
    with open(precision_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_precision :" + str(round(all_precision / len(files), 4)) + "\n" + content)
    f.close()
    with open(recall_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_recall :" + str(round(all_recall / len(files), 4)) + "\n" + content)
    f.close()
    with open(f1_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_f1 :" + str(round(all_f1 / len(files), 4)) + "\n" + content)
    f.close()

def all_poly_IoU(image_path,label_path,poly_label_path,save_path):
    all_iou_txt_path = os.path.join(save_path, "complete_ske_all_poly_IoU.txt")
    complete_ske_path=os.path.join(save_path,"complete_ske")
    sum_IoU = 0
    iou_txt = open(all_iou_txt_path, "w")
    files = os.listdir(image_path)
    for file in files:
        poly_label_name = os.path.join(poly_label_path, file)
        complete_ske_name = os.path.join(complete_ske_path, file)

        # fillPoly
        res_poly=cv2.imread(complete_ske_name,0)
        ske_contours, ske_hierarchy = cv2.findContours(res_poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp_num = 0
        for i in range(len(ske_contours)):
            if ske_hierarchy[0][i][3] != -1:
                cv2.fillPoly(res_poly, [ske_contours[i]], 255)
                if (i + 1 < len(ske_contours) and ske_hierarchy[0][i - 1][3] != -1):
                    if (ske_hierarchy[0][i + 1][3] != ske_hierarchy[0][i][3] and temp_num == 1):
                        cv2.fillPoly(res_poly, [ske_contours[i]], 0)
                        temp_num = 0
                    elif (ske_hierarchy[0][i + 1][3] != ske_hierarchy[0][i][3] and temp_num == 0):
                        temp_num = 1
            else:
                temp_num = 0

        poly_label = cv2.imread(poly_label_name, 0)
        overLap = np.where((poly_label > 0) & (res_poly > 0))[0].size
        u1 = np.where((poly_label > 0) & (res_poly == 0))[0].size
        u2 = np.where((poly_label == 0) & (res_poly > 0))[0].size

        sum = overLap + u1 + u2
        if sum != 0:
            IoU = overLap / sum
        else:
            IoU = 0
        sum_IoU += IoU
        iou_txt.write(file + " :" + str(IoU) + "\n")
    iou_txt.close()
    with open(all_iou_txt_path, "r+")as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Ave_IoU :" + str(round(sum_IoU / len(files), 4)) + "\n" + content)
    f.close()

if __name__ == "__main__":
    image_path=r"F:\zhang_weak\test2\image"
    label_path=r"F:\zhang_weak\test2\label"
    poly_label_path=r"F:\zhang_weak\test2\poly_label"
    save_path=r"F:\zhang_weak\test2"
    all_poly_IoU(image_path,label_path,poly_label_path,save_path)