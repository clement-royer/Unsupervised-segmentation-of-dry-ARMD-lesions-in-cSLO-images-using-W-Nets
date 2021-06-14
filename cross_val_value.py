#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:46:00 2021

@author: thesard
"""
import torch
from torchvision import transforms
import os
import numpy as np
from config import Config
from model import WNet
from skimage.metrics import adapted_rand_error
import statistics

toTensor   = transforms.ToTensor()
path2 = "/media/thesard/DATA/stage_data_clement/ProjetClement/differences_img/"
path3 = "/media/thesard/DATA/stage_data_clement/ProjetClement/Smoothed_img_/"
path4 = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_img/"
path_masks = "/media/thesard/DATA/stage_data_clement/ProjetClement/MATFILES-manu/"
path_gt= "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_gt/"

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
    res = []
    k=0
    for subdir, dirs, files in os.walk(path):
        if k == 0 :
            res.append(files)
        else :
            break
    res = res[0]
    return res

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

nb_cross = 20


config = Config()

index_crossval = 3


mode = 'test'

if mode == 'test':
    

    path_img_test =  "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval/"+str(index_crossval)+"/test/"
    path_gt_test ="/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval_gt/"+str(index_crossval)+"/test/"
elif mode == 'train':
    path_img_test =  "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval/"+str(index_crossval)+"/train/"
    path_gt_test = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval_gt/"+str(index_crossval)+"/train/"
    
imgs_path = get_imgs_path(path_img_test)


path_model =  get_imgs_path("./crossval_model/")
path_model = path_model[index_crossval]

model = WNet()
autoencoder = torch.load(path_model+"model.config")

model_dict = autoencoder.state_dict()
new_model_dict = model.state_dict()

model_dict = {k: v for k, v in model_dict.items() if k in new_model_dict.keys()}
new_model_dict.update(model_dict)
model.load_state_dict(new_model_dict)


model.eval()
    
F1_tot,precision_tot,recall_tot = 0,0,0
std_F1,std_precision,std_recall = [],[],[]
for img_path in imgs_path:
    imgs = get_files(img_path)
    
    F1,precision,recall = 0,0,0
    t = 0
    for img in imgs:
        #print(img_path+img)
        
 
        with open(img_path+img, 'rb') as f:
            image = Image.open(f).convert('L')  
             
        image = toTensor(image).squeeze(0)
        
       # image = torch.tensor(cv2.imread(img_path+img,cv2.IMREAD_GRAYSCALE)).float()
        path_ground_truth = path_gt_test + img_path[-15:]+img  
        
        with open(path_ground_truth, 'rb') as f:
            ground_truth = Image.open(f).convert('L')  
             
        ground_truth = toTensor(ground_truth).squeeze(0)*255
        #ground_truth = torch.tensor(cv2.imread(path_ground_truth,cv2.IMREAD_GRAYSCALE))
        
        
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(ground_truth)
        # plt.show()
        
        segmentation = model.forward_encoder(image.unsqueeze(0).unsqueeze(0))
        segmentation = torch.argmax(segmentation,1)
        
        #Reverse
        segmentation=2-segmentation
        #
        
        # Class index
        segmentation[segmentation != 2] = 0
        segmentation[segmentation != 0] = 255
        #   

       
          
        n = ground_truth.shape[0]*ground_truth.shape[1]
        
        #Count background in evaluation
        segmentation[segmentation == 0] = 1
        ground_truth[ground_truth == 0] = 1
        #
         
        ###
        if t==0:
            previous_seg = segmentation.clone().detach() 
            t += 1
        else:
            segmentation = segmentation+ previous_seg
            segmentation[segmentation==510]=255
            segmentation[segmentation==2]=1
            segmentation[segmentation==256]=255
            
            previous_seg = segmentation.clone().detach() 
            t +=1
         ### 
         
       # plt.imshow(segmentation.squeeze(0))
       # plt.show()
        
        error_i, precision_i, recall_i = adapted_rand_error(np.array(ground_truth.view(n)).astype(int), np.array(segmentation.squeeze(0).view(n)).astype(int))
        k=0
        if torch.sum(ground_truth).item()==0:
            k +=1
        else: 
            F1 += (1-error_i)
            precision += precision_i
            recall += recall_i
            
            std_F1.append(1-error_i)
            std_precision.append(precision_i)
            std_recall.append(recall_i)
    F1,precision,recall =F1/(len(imgs)-k),precision/(len(imgs)-k),recall/(len(imgs)-k)
    print( "-", F1,precision, recall, "-")
    
    F1_tot += F1
    precision_tot += precision
    recall_tot += recall
F1_tot,precision_tot,recall_tot = F1_tot/len(imgs_path),precision_tot/len(imgs_path),recall_tot/len(imgs_path)


print( "--", F1_tot,precision_tot, recall_tot, "--")
print( "==", statistics.stdev(std_F1),statistics.stdev(std_precision), statistics.stdev(std_recall), "==")
print(index_crossval, mode)