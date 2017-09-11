import os
import utils.load_image as load_image
from matplotlib import pyplot
from scipy.misc import toimage
import numpy as np
image_root='/home/junwon/smt-data/images/SOD6'


def main():
    x_train=[]
    for image in os.listdir(image_root):
        image_path = os.path.join(image_root,image)
        ic = load_image.image_class()
        ic.get_image(image_path)
        ic.resize_image(224)
        x_train.append(ic.resized_image)

    x_train = np.array(x_train)
    print x_train.shape
    show(x_train)
def show(data):
    for i in range(0,4):
        pyplot.subplot(220 + 1 +i)
        pyplot.imshow(toimage(data[i]))
    pyplot.show()


main()
