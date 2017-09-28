from __future__ import print_function
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
import logging
import cv2
from datetime import datetime
import os
import skimage.transform
import time
import smt_process.read_component_db as read_component_db


def make_dcgan_sym(ngf, ndf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
    d5 = mx.sym.Flatten(d5)

    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')
    return gout, dloss

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    print("old shape",X.shape)
    X = X.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64,64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 64, 64))
    X = np.tile(X, (1, 3, 1, 1))
    #X_train = X[:60000]
    #X_test = X[60000:]
    X_train = X[:10000]
    X_test = X[1000:2000]
    print("X_train.shape:",X_train.shape)
    print("X_test.shape:",X_test.shape)
    return X_train, X_test

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

class ImagenetIter(mx.io.DataIter):
    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec = path,
            data_shape  = data_shape,
            batch_size  = batch_size,
            rand_crop   = True,
            rand_mirror = True,
            max_crop_size = 256,
            min_crop_size = 192)
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data = data * (2.0/255.0)
        data -= 1
        return [data]

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
   # buf[sy:sy+shape[1], sx:sx+shape[0], :] = img
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img.reshape((64,64,3))



def visual(title, X):
    assert len(X.shape) == 4
    print("before transpose:",X.shape)
    
    plt.imshow(X[0].reshape(64,64,3))
#    plt.title("before transpose") 
#    plt.show()
#   
#    X = X.transpose((0, 2, 3, 1))
#    plt.imshow(X[0].reshape(64,64,3))
#    plt.title("after tranpose")
#    plt.show()
#    

#    print("after transpose:",X.shape)
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
#    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
#    for i, img in enumerate(X):
#        fill_buf(buff, i, img, X.shape[1:3])
    buff = np.zeros((int(n*X.shape[2]), int(n*X.shape[3]), int(X.shape[1])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[2:4])


    
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    plt.imshow(buff)
    plt.title(title)
    plt.show()
#
def get_smtdata():
    image_root='/home/junwon/smt-data/images/images_bgr/R0402'
    img_list=[]
    for image in os.listdir(image_root):
        image_path = os.path.join(image_root,image)
        img = plt.imread(image_path)
        #img = skimage.transform.resize(img,[224,224])
        img_list.append(img)

    X = np.asarray([cv2.resize(x,(64,64)) for x in img_list])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    num_img = X.shape[0]
    X = np.reshape(X,(num_img,3,64,64))
    print("X.shape:",X.shape)
    return X
#
#    return np.array(img_list)



#    os.system("python /home/junwon/mxnet_code/mxnet/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --test-ratio=0.2 data/images-smt ~/smt-data/images/images_bgr_r0402")
#    os.system("python /home/junwon/mxnet_code/mxnet/tools/im2rec.py --num-thread=4 --pass-through=1 data/images-smt ~/smt-data/images/images_bgr_r0402")
#
#    image_dim=224
#    train_iter = mx.image.ImageIter(batch_size=batch_size,data_shape=(3,image_dim,image_dim),
#                path_imgrec="/home/junwon/mxnet_code/smt_project/data/images-smt.rec",
#                path_imgidx="/home/junwon/mxnet_code/smt_project/data/images-smt.idx")
#
#    train_iter.reset()
#    val_iter = mx.image.ImageIter(batch_size=batch_size,data_shape=(3,image_dim,image_dim),
#                    path_imgrec="/home/junwon/mxnet_code/smt_project/data/images-smt.rec",
#                    path_imgidx="/home/junwon/mxnet_code/smt_project/data/images-smt.idx")
#
#    val_iter.reset()
#
#    return train_iter, val_iter
    #batch = train_iter.next()
    #data = batch.data[0]
    # X_train = data
    #X_test = data
    #return X_train, X_test

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
    test_data = rand_images[:train_size]
    test_label = rand_labels[:train_size]
    #test_data =  rand_images[train_size:]
    #test_label = rand_labels[train_size:]
    print "train_size:",len(train_data)
    print "test_size:", len(test_data)
    return num_classes, train_data, train_label, test_data, test_label




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # =============setting============
    dataset = 'smtdata'
    imgnet_path = './train.rec'
    ndf = 64
    ngf = 64
    nc = 3
    batch_size = 64
    Z = 100
    lr = 0.0002
    beta1 = 0.5
    #ctx = mx.gpu(0)
    ctx = mx.cpu()
    check_point = False

    symG, symD = make_dcgan_sym(ngf, ndf, nc)
    #mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
    #mx.viz.plot_network(symD, shape={'data': (batch_size, nc, 64, 64)}).view()

    # ==============data==============
    if dataset == 'mnist':
        X_train, X_test = get_mnist()
        train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size)
        print(X_train.shape)

    elif dataset == 'imagenet':
        train_iter = ImagenetIter(imgnet_path, batch_size, (3, 64, 64))
    elif dataset == 'smtdata':
        print("smtdata")
        #X = get_smtdata()
        #train_iter = mx.io.NDArrayIter(X,batch_size=batch_size)
        parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
        group = parser.add_argument_group()
        group.add_argument("-p", "--k3dpath", default=os.path.expanduser("some file"), help="Path to the k3d file")
        group.add_argument("-t", "--table", default="", help="Path to the mapping table")
        group.add_argument("-g", "--gpu", default="", help="GPU?(True/False)")
        args = parser.parse_args()

        use_gpu = False
        DC = data_class.DataClass()
        
        # set template path
        if args.table != "":
            if os.path.exists(os.path.expanduser(args.table)):
                print "loading table"
                DC.set_template_path(os.path.expanduser(args.table))
        
        # set k3d data path
        if args.k3dpath != "":
            if os.path.exists(os.path.expanduser(args.k3dpath)):
                print "loading k3d files"
                DC.set_data_path(os.path.expanduser(args.k3dpath))




    rand_iter = RandIter(batch_size, Z)
    label = mx.nd.zeros((batch_size,), ctx=ctx)

    # =============module G=============
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })
    mods = [modG]

    # =============module D=============
    modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    print("hi")
    print(train_iter.provide_data)

    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })
    mods.append(modD)


    # ============printing==============
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print('Training...')
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
    num_epoch=500
    # =============train===============
    for epoch in range(num_epoch):
        train_iter.reset()

        if epoch % 2 ==0:
            print("epoch:",epoch,"-",time.strftime("%H:%M:%S"))
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()

            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            label[:] = 0
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            #modD.update()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update discriminator on real
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            modD.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update generator
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD.get_input_grads()
            modG.backward(diffD)
            modG.update()

            mG.update([label], modD.get_outputs())


            if mon is not None:
                mon.toc_print()

            t += 1
            if epoch == num_epoch-1 and t % 2 == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get())
                mACC.reset()
                mG.reset()
                mD.reset()
                print("gout")
                visual('gout', outG[0].asnumpy())
                diff = diffD[0].asnumpy()
                diff = (diff - diff.mean())/diff.std()
                #print("diff")
                #visual('diff', diff)
                print("data")
                visual('data', batch.data[0].asnumpy())

        if check_point:
            print('Saving...')
            modG.save_params('%s_G_%s-%04d.params'%(dataset, stamp, epoch))
            modD.save_params('%s_D_%s-%04d.params'%(dataset, stamp, epoch))
