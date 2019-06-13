"""Functions for image data visualization."""

import matplotlib.pyplot as plt # for showing image
import matplotlib.image as mpimg # for reading image
import numpy as np
import math

"""
img = mpimg.imread('lena.png') #shape [M,N,3 or 4]
#原图
plt.subplot(231)
plt.imshow(img)
plt.title('origin', loc='right')

#R通道
plt.subplot(232)
img1 = img[:, :, 0]  #shape [M,N]
plt.imshow(img1)
plt.title('r', loc='right')

#G通道
plt.subplot(233)
img2 = img[:, :, 1]  #shape [M,N]
plt.imshow(img2)
plt.title('g', loc='right')

#B通道
plt.subplot(234)
img3 = img[:, :, 2]  #shape [M,N]
plt.imshow(img3)
plt.title('b', loc='right')

#灰度图
plt.subplot(235)
img4 = np.dot(img[..., :3], [0.299, 0.587, 0.114])    #shape [M,N]
plt.imshow(img4, cmap=plt.cm.gray)   #cmap='gray'
plt.title('gray', loc='right')

#灰度图反转
plt.subplot(235)
img4 = np.dot(img[..., :3], [0.299, 0.587, 0.114])
plt.imshow(img4, cmap=plt.cm.gray_r)
plt.title('gray', loc='right')

plt.show()
"""


def data_vis(images, labels):
    image_shape = np.shape(images)
    images = np.reshape(images, [image_shape[0], image_shape[1], image_shape[2]])
    total_num = images.shape[0]
    num_per_axis = math.sqrt(total_num)
    num_per_axis = math.ceil(num_per_axis)

    fig, axes = plt.subplots(num_per_axis, num_per_axis)
    count = 0
    for i in range(num_per_axis):
        for j in range(num_per_axis):
            axes[i, j].imshow(images[count]) #, cmap=plt.cm.gray)
            axes[i, j].axis('off')
            axes[i, j].set_title(labels[count], loc='right')
            count += 1
            if count == total_num:
                break
        if count == total_num:
            break
    plt.show()
