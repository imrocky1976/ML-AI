"""Functions for image data visualization."""

import matplotlib.pyplot as plt  # for showing image
# import matplotlib.image as mpimg  # for reading image
import math

"""
lena = mpimg.imread('lena.jpg') # reading lena.jpg
# here lena is a np.array with shape [height, width, depth]
plt.subplot(121)
plt.imshow(lena)  # show image
plt.axis('off')  # hidden axis

plt.subplot(122)
plt.imshow(lena)  # show image
plt.axis('off')  # hidden axis
plt.show()
"""


def data_vis(images, pred_labels=None, real_labels=None):
    total_num = images.shape[0]
    num_per_axis = math.sqrt(total_num)
    num_per_axis = math.ceil(num_per_axis)

    fig, axes = plt.subplots(num_per_axis, num_per_axis)
    count = 0
    for i in range(num_per_axis):
        for j in range(num_per_axis):
            axes[i, j].imshow(images[count])
            axes[i, j].axis('off')
            if pred_labels is not None and real_labels is not None:
                axes[i, j].set_title("pred:%s,real:%s" % (pred_labels[count], real_labels[count]))
            elif pred_labels is None and real_labels is not None:
                axes[i, j].set_title("real:%s" % real_labels[count])
            elif pred_labels is not None and real_labels is None:
                axes[i, j].set_title("pred:%s" % pred_labels[count])
            count += 1
            if count == total_num:
                break
        if count == total_num:
            break
    plt.show()
