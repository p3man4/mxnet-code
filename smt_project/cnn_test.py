import mxnet as mx
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn import preprocessing
import smt_process.read_component_db as read_component_db

SEED=7

def get_dataset():
    images,labels = get_data()

    # shuffle images and lables
    np.random.seed(SEED)
    assert len(images) == len(labels)
    rand_index = np.arange(len(images))
    np.random.shuffle(rand_index)
   
    rand_images = images[rand_index]
    rand_labels = labels[rand_index]

    train_size = int(len(rand_images) * 0.8)
    test_size =  len(rand_images) - train_size

    train_data = rand_images[:train_size]
    train_label = rand_labels[:train_size]
    test_data =  rand_images[train_size:]
    test_label = rand_labels[train_size:]
    
    return train_data, train_label, test_data, test_label


def get_smtdata():
    data_root='/home/junwon/smt-data/images/images_bgr/'

    cls_id=0
    cls_id_table={}
    data=[]
    label=[]
    for cls_dir in os.listdir(image_root):
        cls_id_table[cls_dir]=cls_id
        cls_dir_path = os.path.join(image_root,cls_dir)
        for image in  os.listdir(cls_dir_path):
            image_path = os.path.join(cls_dir_path,image)
            img  = plt.imread(image_path)
            img  = cv2.resize(img,(64,64))
            print img.shape
    #        img  = preprocessing.scale(img)
    #        print img.shape
            data.append(img)
            label.append(cls_id)
        cls_id = cls_id + 1
    
    X = np.asarray(data)
    #X = np.asarray(cv2.resize(x,(64,64)) for x in data)
    X = X.astype(np.float32)/(255.0/2) - 1.0
    num_img = X.shape[0]

    X = np.reshape(X,(num_img,3,64,64))

   # img_list=[]
   # for image in os.listdir(image_root):
   #     image_path = os.path.join(image_root,image)
   #     img = plt.imread(image_path)
   #     #img = skimage.transform.resize(img,[224,224])
   #     img_list.append(img)

   # X = np.asarray([cv2.resize(x,(64,64)) for x in img_list])
   # X = X.astype(np.float32)/(255.0/2) - 1.0
   # num_img = X.shape[0]
   # X = np.reshape(X,(num_img,3,64,64))
    print("X.shape:",X.shape)
    return X
def get_mnist():
    mnist = mx.test_utils.get_mnist()
    print 'train shape:',mnist['train_data'].shape
    print 'test shape:',mnist['test_data'].shape

    train_data = mnist['train_data'][:1000]
    train_label = mnist['train_label'][:1000]

    test_data = mnist['test_data'][:100] # changed
    test_label = mnist['test_label'][:100] # changed

    return train_data,train_label,test_data,test_label
 
def main():

 #   train_data,train_label,test_data,test_label = get_mnist()
 #   train_data,train_label,test_data,test_label = get_smtdata()
    train_data, train_label, test_data, test_label = get_dataset()
    data = mx.sym.var('data')

    batch_size = 10
    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)


    conv1 = mx.sym.Convolution(data=data,kernel=(5,5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type='tanh')
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))


    conv2 = mx.sym.Convolution(data=pool1,kernel=(5,5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type='tanh')
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))


    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")

    fc2 = mx.sym.FullyConnected(data=tanh3,num_hidden=10)

    lenet = mx.sym.SoftmaxOutput(data=fc2,name='softmax')

    lenet_model = mx.mod.Module(symbol=lenet,context = mx.cpu())
    print 'training begin'
    lenet_model.fit(train_iter,
            eval_data=val_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate':0.1},
            eval_metric='acc',
            batch_end_callback= mx.callback.Speedometer(batch_size,100),
            num_epoch=10)
    print 'training end'
    test_iter = mx.io.NDArrayIter(test_data, None, batch_size)
    prob = lenet_model.predict(test_iter)
    test_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)
    # predict accuracy for lenet
    acc = mx.metric.Accuracy()
    lenet_model.score(test_iter, acc)
    print(acc)
    #assert acc.get()[1] > 0.98

#get_smtdata()
train_data,train_label,test_data,test_label = get_mnist()
#print "train_data:",train_data
#print "train_label:",train_label
print "train_data:",train_data.shape
print "train_label:",train_label.shape


# main()
