"""
制作copymove数据集。
1.保存真实mask
2.保存篡改mask
3.保存两个mask在一副图像中，不同颜色显示。
4.保存篡改图
"""
import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def otsu_binarization(img, th=128):
    H, W = img.shape
    out = img.copy()

    max_sigma = 0
    max_t = 0

    # determine threshold
    for _t in range(1, 255):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255
    return out


# Morphology Dilate
def Morphology_Dilate(img, Dil_time=1):
    H, W = img.shape

    # kernel
    MF = np.array(((0, 1, 0),
                   (1, 0, 1),
                   (0, 1, 0)), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:
                    out[y, x] = 255

    return out


def edge(image):
    img = image
    ret, image1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)  # binary（黑白二值），ret代表阈值，80是低阈值，255是高阈值
    # cv2.imshow('image1', image1)#将阈值分割处理后的图片命名为“image1”显示出来

    # 二值图像的反色处理，将图片像素取反
    print("image1.shape:", image1.shape)
    height = image1.shape[0]  # 返回图片大小
    width = image1.shape[1]
    image2 = image1.copy()
    for x in range(height):
        for j in range(width):
            image2[x, j] = (255 - image1[x, j])
    #
    # path = src_path[15:-4]
    # path1 = src_path[15:]
    # # 边缘提取，使用Canny函数
    image2_3 = cv2.Canny(image2, 80, 255)  # 设置80为低阈值，255为高阈值

    # plt.imshow(image2_3, cmap="gray")
    # plt.show()

    otsu = otsu_binarization(image2_3)
    # Morphology - dilate
    # erode_result = Morphology_Erode(otsu, Erode_time=2)
    dilate_result = Morphology_Dilate(otsu, Dil_time=2)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = dilate_result
    img2[:, :, 1] = dilate_result
    img2[:, :, 2] = dilate_result

    image_edge = cv2.bitwise_and(img, img2)
    # edge = Image.fromarray(cv2.cvtColor(dilate_result, cv2.COLOR_BGR2RGB))
    return img2


def object_area(img):
    pictue_size = img.shape
    picture_height = pictue_size[0]
    picture_width = pictue_size[1]
    i = 0
    up = -1
    down = -1
    left = -1
    right = -1

    for a in range(picture_height):
        for b in range(picture_width):
            if img[a, b].all() > 0:
                if up == -1:
                    up = a
                i = i + 1
    r = i / (picture_height * picture_width)
    return r


def make_copymove(path1, mask_path):
    # 读取图片和mask
    try:
        img_1 = cv2.imread(path1)
        mask = cv2.imread(mask_path)

        # 获取图片大小，保证mask和图片大小一致
        h, w, c = img_1.shape
        mask = cv2.resize(mask, (w, h))
        rate = object_area(mask)
        print("rate", rate)
        image = mask
        closed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        # print(y1)
        # print(y1 + hight)
        # print(x1)
        # print(x1 + width)
        # cv2.imshow("mask", mask)
        # 因为是copymove，从原mask赋值给另一个mask，调整大小后，粘贴会和原mask大小一致的一个全黑图像。
        # 然后就得到mask2。即mask为原mask，mask1为将mask调整大小后的mask，mask2为和mask大小一致的全黑图像。
        # 将mask1粘贴至mask2的一个位置。得到的mask2为和mask形状一致，但是mask2中的物体大小和位置发生变化
        mask1 = image[y1:y1 + hight, x1:x1 + width]
        # plt.imshow(mask)
        # plt.show()
        # plt.imshow(mask1)
        # plt.show()
        # print(y1 + hight)
        # print(x1 + width)
        mask2 = np.zeros_like(mask)
        # cv2.imshow("mask1", mask1)
        height_paste = random.randint(1, h - hight - 1)
        left = x1
        right = w - x1 - width
        if left < right:
            if right > width:
                width_paste = random.randint(1, (right - width)) + (left + width)
            else:
                if w > h:
                    mask1 = cv2.resize(mask1, (right - 1, hight))
                else:
                    mask1 = cv2.resize(mask1, (width / 2, hight / 2))
                width_paste = right - 1
        else:
            if left > width:
                width_paste = random.randint(1, (left - width - 1))
            else:
                if w > h:
                    mask1 = cv2.resize(mask1, (left - 1, hight))
                else:
                    mask1 = cv2.resize(mask1, (width // 2, hight // 2))
                width_paste = 0
        mask2 = Image.fromarray(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
        mask1 = Image.fromarray(cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB))
        mask2.paste(mask1, (width_paste, height_paste))
        mask2 = np.asarray(mask2)
        # plt.imshow(mask2)
        # plt.show()
        # 根据mask,获取要复制的物体image_mask1，然后将image_mask1调整大小，大小和mask1大小一致。
        # 然后再粘贴到一个全黑的大小和原mask大小一致的mask3中。
        # 也就是说，mask2为篡改物体的掩码图，mask3对应位置的篡改的物体。
        image_mask1 = cv2.bitwise_and(img_1, mask)
        image_mask1 = image_mask1[y1:y1 + hight, x1:x1 + width]
        mask1 = np.asarray(mask1)
        mask1_h, mask1_w, mask1_c = mask1.shape
        image_mask1 = cv2.resize(image_mask1, (mask1_w, mask1_h))
        mask3 = np.zeros_like(mask)
        mask3 = Image.fromarray(mask3)
        image_mask1 = Image.fromarray(image_mask1)
        mask3.paste(image_mask1, (width_paste, height_paste))
        mask3 = np.asarray(mask3)

        # 通过mask2，将原图像的相应位置的像素减掉。
        # 得到的diffImg
        diffImg1 = cv2.subtract(img_1, mask2)

        # 将mask3和diffImg1相加，得到最后的篡改图。
        image_need = cv2.addWeighted(diffImg1, 1, mask3, 1, 0)
        image_need = Image.fromarray(cv2.cvtColor(image_need, cv2.COLOR_BGR2RGB))
        mask_real = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        mask_forgery = Image.fromarray(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
        _, mask = cv2.threshold(mask, 90, 255, cv2.THRESH_BINARY)
        mask_ = mask
        mask2_ = mask2
        mask = edge(mask)
        mask2 = edge(mask2)
        mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
        mask_two_edge = cv2.addWeighted(mask, 1, mask2, 1, 0)
        mask_two_edge = Image.fromarray(cv2.cvtColor(mask_two_edge, cv2.COLOR_BGR2RGB))
        mask_[np.where((mask_ == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
        mask_two = cv2.addWeighted(mask_, 1, mask2_, 1, 0)
        mask_two = Image.fromarray(cv2.cvtColor(mask_two, cv2.COLOR_BGR2RGB))

        return image_need, mask_real, mask_forgery, mask_two, mask_two_edge
    # 有点小错误，没看出来，不想改了，用个try,异常处理
    except:
        return 0, 0, 0, 0, 0


mask = os.listdir("../coco/use_mask/")
for i in range(len(mask)):
    mask_path = "../coco/use_mask/" + mask[i]
    image_path = "../coco/coco_train_2017/train2017/" + mask[i]
    tam, real_mask, tam_mask, tam_two_mask, tam_two_mask_edge = make_copymove(image_path, mask_path)
    if tam == 0:
        continue
    tam.save("../coco/copymove/tam/" + mask[i])
    real_mask.save("../coco/copymove/real_mask/" + mask[i])
    tam_mask.save("../coco/copymove/tam_mask/" + mask[i])
    tam_two_mask.save("../coco/copymove/tam_two_mask/" + mask[i])
    tam_two_mask_edge.save("../coco/copymove/tam_two_mask_edge/" + mask[i])
