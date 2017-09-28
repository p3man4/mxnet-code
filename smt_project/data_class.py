####################################################################
#
# (c) 2017 Koh YOung Research America (KYRA)
#
# Proprietary and confidential
#
# retrieve data set from k3d files 
#
####################################################################
import numpy as np
import sys
import os
import smt_process.detect_class as detect_class
import smt_process.template_2_part as template_2_part
import cv2
import matplotlib.pyplot as plt

class DataClass(object):
    """
    Container class for  data related information
    """
    def __init__(self):
        self.data_path = None
        self.template_path = None
        self.img_resize = None
    
    def set_template_path(self,template_path):
        self.template_path = template_path

    def set_data_path(self,data_path):
        self.data_path = data_path

    def set_img_resize(self,img_resize):
        self.img_resize = int(img_resize)

    def load_k3d(self):
        part_template_dict = template_2_part.build_table(self.template_path)
        template_images_dict={}
        template_id_dict={}
        id_images_dict={}
        images=[]
        labels=[]
        for subd in os.listdir(self.data_path):
            print subd, "is processing"
            subd_path = os.path.join(self.data_path,subd)
            # pass if subd is not subdirectory
            if not os.path.isdir(subd_path): 
                continue
            for idx, f in enumerate(os.listdir(subd_path)):
                # pass if f is txt file
                if f.endswith('txt'): 
                    continue
                f_name = f.split('.')[0]
                fd = open(os.path.join(subd_path,f))
                DC = detect_class.ComponentDetector()
                k3d_dict = DC.parse_k3d(fd.read())
                fd.close()

                part_nm = k3d_dict['component_id']
                template_nm=None
                try:
                    template_nm= part_template_dict[part_nm][0]
                #    print 'idx',idx, ":", part_nm,":",template_nm
                except:
                    print "template_nm error"
                    raise
                if not template_nm in template_images_dict.keys():
                    template_images_dict[template_nm] = [k3d_dict['img_bgr']]
                else:
                    template_images_dict[template_nm].append(k3d_dict['img_bgr'])

        num_classes= len(template_images_dict.keys())
        # map template nm to id & 
        # map template id to images
        key_id = 1
        for key, value in template_images_dict.iteritems():
            #print 'key:',key, 'len(value):',len(value)
            template_id_dict[key]=key_id
            id_images_dict[key_id] = value
            key_id = key_id + 1
        
        id_template_dict = {v:k for k,v in  template_id_dict.iteritems()}

        for key_id, imgs in id_images_dict.iteritems():
            for i,img in enumerate(imgs):
                if i <= 3:
                    template_nm = id_template_dict[key_id]
                # resize, normalize, zero mean, one variance                
                img= cv2.resize(img,(self.img_resize,self.img_resize))
                img = img /255.0
                img = (img - np.mean(img))/np.std(img)
                labels.append(key_id)
                images.append(img)

        images2 = np.array(images)
        labels2 = np.array(labels)
        
        print "num classes:", num_classes
        return num_classes,images2,labels2

if __name__ == "__main__":
    img_resize=56
    template_path="/home/swchoi/smt-project/SMT-v2/detect_part/training_data/Template2Part_Map.csv"
    data_path="/home/swchoi/smt-data/Train_0818_extra"
    img_resize=56

    DC = DataClass()
    DC.set_template_path(template_path)
    DC.set_data_path(data_path)
    DC.set_img_resize(img_resize)

    num_classes, images,labels = DC.load_k3d()
    print "num_classes:", num_classes
    print "images.shape:",images.shape
    print "lablels.shape:",labels.shape
    print "num_classes:",num_classes
