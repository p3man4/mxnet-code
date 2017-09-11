import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import mxnet as mx
from sklearn.datasets import fetch_mldata
import cv2
import os
def main():
    print  "main"

    data="mnist"
    common(data)





def mnist():
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234)
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000,28,28))
    X = np.asarray([cv2.resize(x,(64,64)) for x in X])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000,1,64,64))
    X = np.tile(X,(1,3,1,1))
    X_train = X[:1000]
    return X_train


def smt():
    image_root='/home/junwon/smt-data/images/images_bgr/R0402'
    img_list=[]
    for image in os.listdir(image_root):
        image_path = os.path.join(image_root,image)
        img  = plt.imread(image_path)
        img_list.append(img)

    X = np.array([cv2.resize(x,(64,64)) for x in img_list])
    X  = X.astype(np.float32)/(255.0/2) - 1.0
    num_img = X.shape[0]
    #plt.imshow(X[0])
    #plt.show()
    X = np.reshape(X,(num_img,3,64,64))
    plt.imshow(X[0].reshape((64,64,3)))
    plt.show()
    X_train = X[:1000]
    return X_train

def common(data):

    if data=="mnist":
        X_train = mnist()
    else:
        X_train = smt()

    batch_size=1
    train_iter = mx.io.NDArrayIter(X_train,batch_size=batch_size)
    train_iter.reset()

    batch = list(enumerate(train_iter))[0][1]
    p = batch.data[0].asnumpy()
    print p.shape

    if data=="mnist":
        p = p.transpose((0,2,3,1))

#    print p.shape
#    plt.imshow(p[0])
    for i,img in enumerate(p):
        print i,",", img.shape

        if data=="smtdata":
            plt.imshow(img.reshape((64,64,3)))
        else:
            plt.imshow(img)
        plt.show()
main()
