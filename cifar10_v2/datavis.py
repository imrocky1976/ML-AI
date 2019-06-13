"""Functions for image data visualization."""

import matplotlib.pyplot as plt  # for showing image
# import matplotlib.image as mpimg  # for reading image
import math

def data_vis(images, labels):
    assert images.ndim == 3 or images.ndim == 4, 'Input images dim must be 1 or 4'
    if images.ndim == 3:
        _, ax = plt.subplots()
        ax.set_title(labels)
        ax.imshow(images)
    elif images.ndim == 4:
        total_num = images.shape[0]
        num_per_axis = math.sqrt(total_num)
        num_per_axis = math.ceil(num_per_axis)

        fig, axes = plt.subplots(num_per_axis, num_per_axis)
        count = 0
        for i in range(num_per_axis):
            for j in range(num_per_axis):
                axes[i, j].imshow(images[count])
                axes[i, j].set_title(labels[count])
                count += 1
                if count == total_num:
                    break
            if count == total_num:
                break
        plt.show()


def show_image(img1, img2, lbl):
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title(lbl)
    axes[0].imshow(img1)
    axes[1].set_title(lbl)
    axes[1].imshow(img2)
    plt.show()


def show_one_image(img, lbl):
    fig, ax = plt.subplots()
    ax.set_title(lbl)
    ax.imshow(img)
    plt.show()
