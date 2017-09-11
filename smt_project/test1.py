import smt_process.detect_class as detect_class
import os
import numpy as np
import matplotlib.pyplot as plt



from scipy import ndimage

DATA_PATH='/home/junwon/smt-data/Train_All/'

f_path=os.path.join(DATA_PATH,'15um_Train/R621.k3d')

f = open(f_path,'r')
DC = detect_class.ComponentDetector()
k3dfile = DC.parse_k3d(f.read())
f.close()

k3d_image = k3dfile['img_bgr']
print k3dfile.keys()

