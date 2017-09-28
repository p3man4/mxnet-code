###########################################################################
#
# (c) 2017 Koh Young Research America (KYRA)
#
# Proprietary and confidential
#
# model zoo 
#
###########################################################################

import mxnet as mx

# define shallow_cnn
def shallow_cnn(data,num_classes):
    conv1 = mx.sym.Convolution(data=data,kernel=(5,5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type='tanh')
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

    conv2 = mx.sym.Convolution(data=pool1,kernel=(5,5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type='tanh')
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))

    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")

    fc2 = mx.sym.FullyConnected(data=tanh3,num_hidden=num_classes)

    shallow_cnn = mx.sym.SoftmaxOutput(data=fc2,name='softmax')

    return shallow_cnn
       


# more models to come 
