###########################################################################
#
# (c) 2017 Koh Young Research America (KYRA)
#
# Proprietary and confidential
#
# Test shallow CNN with k3d files
#
###########################################################################

import mxnet as mx
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn import preprocessing
import sys
sys.path.append('/home/swchoi/smt-project/SMT-v2/detect_part')
import smt_process.read_component_db as read_component_db
import data_class
import argparse
import model_zoo

SEED=7

def get_dataset(DC):
    num_classes, images,labels = DC.load_k3d()

    # convert NHWC to NCHW
    if images.shape[3] == 3:
        images = images.transpose([0,3,1,2])

    # shuffle images and labeles
    np.random.seed(SEED)
    assert len(images) == len(labels)
    rand_index = np.arange(len(images))
    np.random.shuffle(rand_index)
    
    rand_labels = labels[rand_index]
    rand_images = images[rand_index]

    train_size = int(len(rand_images) * 0.8)
    test_size =  len(rand_images) - train_size
    train_data = rand_images[:train_size]
    train_label = rand_labels[:train_size]
    
    # test = train
    #test_data = rand_images[:train_size]
    #test_label = rand_labels[:train_size]
    
    # test != train
    test_data =  rand_images[train_size:]
    test_label = rand_labels[train_size:]
    print "train_size:",len(train_data)
    print "test_size:", len(test_data)
    return num_classes, train_data, train_label, test_data, test_label


def main(DC,use_gpu):
    # get data    
    num_classes,train_data, train_label, test_data, test_label = get_dataset(DC)

    # some parameters
    batch_size = 10
    data = mx.sym.var('data')
    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)

    # get model
    shallow_model = None
    model = model_zoo.shallow_cnn(data, num_classes)

    if use_gpu:
        print 'gpu mode'
        shallow_model = mx.mod.Module(symbol=model, context = mx.gpu())
    else:
        print 'cpu mode'
        shallow_model = mx.mod.Module(symbol=model, context = mx.cpu())
       
    print 'training starts'
    shallow_model.fit(train_iter,
            eval_data=val_iter,
            optimizer='adam',
            optimizer_params={'learning_rate':0.001},
            eval_metric='acc',
            batch_end_callback= mx.callback.Speedometer(batch_size, 100),
            num_epoch=100)
    print 'training ends'
    test_iter = mx.io.NDArrayIter(test_data, None, batch_size)
    prob = shallow_model.predict(test_iter)
    test_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)

    for bath in test_iter:
        print "batch.test_data",batch.test_data
        print "batch.test_label", batch.test_label

    # predict accuracy for model
    acc = mx.metric.Accuracy()
    shallow_model.score(test_iter, acc)
    print test_iter
    #print shallow_model.predict(test_iter).shape
    print(acc)
    #assert acc.get()[1] > 0.98


if __name__ == "__main__":
    desc = """
    First version of shallow CNN
    """
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument("-p", "--k3dpath", default=os.path.expanduser("some file"), help="Path to the k3d files")
    group.add_argument("-t", "--table", default="", help="Path to the mapping table")
    group.add_argument("-g", "--gpu", default="", help="GPU?(True/False)")
    args = parser.parse_args()
    
    # some fixed parameters
    img_resize=56
    use_gpu=False

    DC = data_class.DataClass()
    
    # resize image for saving computation time
    DC.set_img_resize(img_resize)
    
    # set template path
    if args.table != "":
        if os.path.exists(os.path.expanduser(args.table)):
            print "loading table"
            DC.set_template_path(os.path.expanduser(args.table))
    
    # set k3d data path
    if args.k3dpath !="":
        if os.path.exists(os.path.expanduser(args.k3dpath)):
            print "loading k3d files"
            DC.set_data_path(os.path.expanduser(args.k3dpath))
    
    # set gpu (default:False)
    if args.gpu != "":
        use_gpu = args.gpu
    
    main(DC,use_gpu)
