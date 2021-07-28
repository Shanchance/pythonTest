
import random
import numpy as np
import cv2
from PIL import Image


def make_copymove(path1, mask_path):
    # 读取图片和mask
    img_1 = cv2.imread(path1)
    mask = cv2.imread(mask_path)

    # 获取图片大小，保证mask和图片大小一致
    h, w, c = img_1.shape
    mask = cv2.resize(mask, (w, h))

    # 因为是copymove，从原mask赋值给另一个mask，调整大小后，粘贴会和原mask大小一致的一个全黑图像。
    # 然后就得到mask2。即mask为原mask，mask1为将mask调整大小后的mask，mask2为和mask大小一致的全黑图像。
    # 将mask1粘贴至mask2的一个位置。得到的mask2为和mask形状一致，但是mask2中的物体大小和位置发生变化。
    mask1 = cv2.resize(mask, (200, 200))
    mask2 = np.zeros_like(mask)
    mask2 = Image.fromarray(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
    mask1 = Image.fromarray(cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB))
    mask2.paste(mask1, (50, 50))
    mask2.show()
    mask2 = np.asarray(mask2)

    # 根据mask,获取要复制的物体image_mask1，然后将image_mask1调整大小，大小和mask1大小一致。
    # 然后再粘贴到一个全黑的大小和原mask大小一致的mask3中。
    # 也就是说，mask2为篡改物体的掩码图，mask3对应位置的篡改的物体。
    image_mask1 = cv2.bitwise_and(img_1, mask)
    image_mask1 = cv2.resize(image_mask1, (200, 200))
    mask3 = np.zeros_like(mask)
    mask3 = Image.fromarray(mask3)
    image_mask1 = Image.fromarray(image_mask1)
    mask3.paste(image_mask1, (50, 50))
    mask3 = np.asarray(mask3)

    # 通过mask2，将原图像的相应位置的像素减掉。
    # 得到的diffImg
    diffImg1 = cv2.subtract(img_1, mask2)

    # 将mask3和diffImg1相加，得到最后的篡改图。
    image_need = cv2.addWeighted(diffImg1, 1, mask3, 1, 0)
    cv2.imshow("image_need", image_need)
    cv2.waitKey(0)
    image_need = Image.fromarray(cv2.cvtColor(image_need, cv2.COLOR_BGR2RGB))

    return image_need


make_copymove("../image/im1.jpg", "../image/mask.jpg")
