####################################################################
#
# (c) 2017 Koh YOung Research America (KYRA)
#
# Proprietary and confidential
#
# build learning set from k3d files for CNN
#
####################################################################
import numpy as np
import sys
import os
#sys.path.append("/home/swchoi/smt-project/SMT-v2/detect_part")
import smt_process.detect_class as detect_class
import smt_process.template_2_part as template_2_part
TEMPLATE_FILE="/home/swchoi/smt-project/SMT-v2/detect_part/training_data/Template2Part_Map.csv"
DATA_HOME="/home/swchoi/smt-data/Train_0818"

def main():

    part_template_dict = template_2_part.build_table(TEMPLATE_FILE)
    print part_template_dict.keys()

    template_images_dict={}
    template_id_dict={}
    id_images_dict={}
    for subd in os.listdir(DATA_HOME):
        print "subd:",subd

        subd_path = os.path.join(DATA_HOME,subd)
        # pass if subd is not subdirectory
        if not os.path.isdir(subd_path): 
            continue

        for f in os.listdir(subd_path):
            # pass if f is txt file
            if f.endswith('txt'): 
                continue

            f_name = f.split('.')[0]
            print "f_name:",f
            
            fd = open(os.path.join(subd_path,f))
        
            DC = detect_class.ComponentDetector()

            k3d_dict = DC.parse_k3d(fd.read())
            
            fd.close()

            part_nm = k3d_dict['component_id']
            try:
                print part_nm,":",part_template_dict[part_nm][0]
            except:
                print "does not exit"

            template_nm = part_template_dict[part_nm]

            if not template_nm in template_images_dict.keys():
                template_images_dict[template_nm] = [k3d_dict['img_bgr']]
            else:
                template_images_dict[template_nm].append(k3d_dict['img_bgr'])
        print len(template_images_dict.keys())

        # matching template name to template id
        # group template id and images
        key_id = 1
        for key, value in template_images_dict.iteritems():
            template_id_dict[key]=key_id
            id_images_dict[key_id] = value
            key_id = key_id + 1
        
        images=[]
        labels=[]
        for key, value in id_images_dict.iteritems():
            for img in value:
                labels.append(key)
                images.append(img)

        print "labels:",labels
        return images,labels


            

#def build_part_template_dict():
#    part_template_dict={}
#    
#    with open(TEMPLATE_FILE,'r') as f:
#        next(f)
#        for index,line in enumerate(f):
#            template_nm = line.split(",")[1]
#            template_nm = template_nm.replace("\"","")
#
#            part_nm = line.split(",")[2]
#            part_nm = part_nm.replace("\"","")
#            part_nm = part_nm[0:20]
#            part_template_dict[part_nm] = template_nm
#
#    return part_template_dict



main()
