from __future__ import print_function
from __future__ import division
import pandas as pd
import torch
import numpy as np
from skimage.segmentation import chan_vese
import sys
config_path ="/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/config.py"

sys.path.append(os.path.dirname(os.path.expanduser(config_path)))
from config import Config

import cv2

toTensor   = transforms.ToTensor()

config = Config()


def load_pil_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    
def get_subdir(path):
    res = []
    k=0
    for subdir, dirs, files in os.walk(path):
        if k == 0 :
            res.append(dirs)
        else :
            break
    res = res[0]
    return res


def get_files(path):
    for subdir, dirs, files in os.walk(path):
        return files



def get_imgs_path(path):  
    patients = get_subdir(path)  
    path_patients = []  
    for patient in patients :
        path_patients.append([path+patient])      
    nb_patient = len(path_patients)  
    res = []
    for k in range(nb_patient):
        eyes = get_subdir(path_patients[k][0])
        for eye in eyes :
            res.append(path_patients[k][0] + '/' + eye +'/')  
    return res

path_gts = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_gt/"
path_imgs = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_processed_img/"


list_patients = get_imgs_path(path_imgs)
list_patients_gt = get_imgs_path(path_gts)


dict = {'Patient' : [], 'F1' : [],'Precision' : [],'Recall' : [] }
res = pd.DataFrame(dict)

# ### loop on all data for active contour segmentation
# for k in range(len(list_patients)):
#     img_path = list_patients[k]
#     patient = img_path[-15:]
#     list_imgs = get_files(img_path)
    
#     gt_path = list_patients_gt[k]
    
#     f1, precision, recall = 0,0,0
#     for i in range(len(list_imgs)):
#         img = img_path+list_imgs[i]
#         with open(img, 'rb') as f:
#             img = Image.open(f).convert('L')
#        # img = toTensor(img).float().cuda()

#         gt = gt_path+list_imgs[i]
#         with open(gt, 'rb') as f:
#             gt = Image.open(f).convert('L')
#         gt = toTensor(gt).float()
             
#         image = img_as_float(img)
#         cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
#                        dt=0.5, init_level_set="checkerboard", extended_output=True)
#         segmentation = torch.tensor(cv[0].astype(int))
        
        
#         segmentation[segmentation == 1 ] = 255
#         segmentation[segmentation == 0 ] = 1
        
        
#         gt[gt == 1 ] = 255
#         gt[gt == 0 ] = 1
        
        
#         n = 256*256
#         error_i, precision_i, recall_i = adapted_rand_error(np.array(gt.view(n)).astype(int), np.array(segmentation.view(n)).astype(int))
        
#         f1 += (1-error_i)
#         precision += precision_i
#         recall += recall_i

#         f, axes = plt.subplots(1, 3, figsize=(48,48))
#         axes[0].imshow(toTensor(img).squeeze(0))
#         axes[1].imshow(gt.squeeze(0).squeeze(0).detach().cpu())
#         axes[2].imshow(cv[0])
#         plt.show()
        
#     f1 = f1/len(list_imgs)
#     precision = precision/len(list_imgs)
#     recall = recall/len(list_imgs)
#     dict = {'Patient' : patient, 'F1' : f1,'Precision' :precision,'Recall' : recall }
#     res = res.append(dict, ignore_index=True)
#     print(f1, precision, recall)

img_to_visualize = ["005_AU_R/IR OG/aligned_6_20171011_mapped.png.png",  "117_ZE_C/IR OG/aligned_5_20120910_mapped.png.png", "016_BL_G/IR OG/aligned_20_20141119_mapped.png.png"]
path_to_store =  "/media/thesard/DATA/stage_data_clement/ProjetClement/kanezaki_example/"
for im in img_to_visualize : 
    image = path_imgs + im
    
    with open(image, 'rb') as f:
        img = Image.open(f).convert('L')
      
             
    image = img_as_float(img)
    cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,dt=0.5, init_level_set="checkerboard", extended_output=True)
    segmentation = torch.tensor(cv[0].astype(int))
    # plt.imshow(segmentation)
    # plt.show()
     
    cv2.imwrite(path_to_store + im, np.array(segmentation))
    print("done")
    
    