import mxnet as mx
import tarfile



fname = mx.test_utils.download(url='http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz', dirname='data', overwrite=False)

tar = tarfile.open(fname)

tar.extractall(path='./data')

tar.close()
