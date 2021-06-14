from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np
import glob
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.io import loadmat

from config_W_net import Config

ImageFile.LOAD_TRUNCATED_IMAGES = True
config = Config()

file_ext = ".jpg"

randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(500)
toTensor   = transforms.ToTensor()
toPIL      = transforms.ToPILImage()

#path = "D:/stage_data_clement/ProjetClement/DMLA-TimeLapse-Align-corrected-auto/"
path = "/media/thesard/DATA/stage_data_clement/ProjetClement/Processed_img/"
path2 = "/media/thesard/DATA/stage_data_clement/ProjetClement/differences_img/"
path3 = "/media/thesard/DATA/stage_data_clement/ProjetClement/Smoothed_img_/"
path4 = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_img/"
path5 = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_processed_img/"
path_masks = "/media/thesard/DATA/stage_data_clement/ProjetClement/MATFILES-manu/"
path_gt= "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_gt/"
path_gt_5= "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_processed_gt/"

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


def get_imgs(imgs_path, index):
    imgs=[]
    
    for subdir, dirs, files in os.walk(imgs_path[index]):
        for file in files:
            imgs.append(imgs_path[index]+ '/' +file)    
    print("nombres d'images : ",len(imgs))    
    cropped_imgs = []
    for i in range(len(imgs)):
        img = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
        cropped_imgs.append(img)
    return cropped_imgs

def get_imgs_masks(imgs_path, masks_path,index):
    #Get mask + initial mask (baseline mask for one image series)
    com = loadmat(masks_path[index] +'/COM.mat')
    #do = loadmat(masks_path[index] + '/DO.mat') 
    initial_mask = com['maskAnalysis']
    masks = com['MASK']
    
    #Get images and apply the initial mask    
    imgs=get_imgs(imgs_path, index)
    for k in range(len(imgs)):
        imgs[k] = imgs[k]*initial_mask
                    
    return imgs, masks


def reverse_segmentation(img):
    if img[0][0]==255 :
        img = 255 - img
    return img



class AutoencoderDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
  
        self.image_list, self.masks_list = self.get_image_list()
        
        

    def __len__(self):
        return min(len(self.image_list),len(self.masks_list))
    def load_pil_image(self, path):
    # open path as file to avoid ResourceWarning
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
    def __getitem__(self, i):
        # Get the ith item of the dataset
        # Random crop
        if self.mode == "train":
            im = self.load_pil_image(self.image_list[i]) # remove centercrop (500)
            i_, j_, h_, w_ = transforms.RandomCrop.get_params(im, output_size=(config.input_size, config.input_size))
            im = TF.crop(im,i_, j_, h_, w_)
            
        if self.mode == "val":
            im = centerCrop_val(self.load_pil_image(self.image_list[i]))
            
        img = toTensor(im)
        if self.mode == "train":
            
            mas = self.masks_list[i]
            
            mas = TF.crop(mas,i_, j_, h_, w_)
            
            
        if self.mode == "val":
            mas = centerCrop_val(toPIL(self.masks_list[i]))
        mask = toTensor(mas)
        return img, mask 

    def get_image_list(self):
        if config.all_image == False :
            if self.mode == "train":
                index = 12
            if self.mode == "val":
                index = 12
            imgs_path = np.array(get_imgs_path(path))
            
            masks_path = np.array(get_imgs_path(path_masks))
            # number 13 can be replaced by its copy (14-16)
            
            indices=np.array([0,1,2,4,5,6,7,8,10,11,12,13,17,18,19,20,21,22])
            
            masks_path = masks_path[indices]
            
            com = loadmat(masks_path[index] +'/COM.mat') 
       
            masks = com['MASK']
               
            image_path = imgs_path[index]
            
            imgs_list = []
            for subdir, dirs, files in os.walk(image_path):
                for file in files:
                    imgs_list.append(image_path+file)
        if config.all_image == True :

            imgs_path = np.array(get_imgs_path(path3))
            
            masks_path = np.array(get_imgs_path(path_masks))
            # number 13 can be replaced by its copy (14-16)
            
            indices=np.array([0,1,2,4,5,6,7,8,10,11,12,13,17,18,19,20,21,22])
            
            masks_path = masks_path[indices]
            
            
            indices2 = np.array([0,1,3,4,5,6,7,8,10,11,12,13,15,16])
            
            imgs_path= imgs_path[indices2]
            masks_path= masks_path[indices2]

             
            imgs_list = []
            masks_grouped_list = []
            masks_list = []
            
            for k in range(len(imgs_path)):
            #image
                image_path = imgs_path[k]
                for subdir, dirs, files in os.walk(image_path):
                    for file in files:
                        imgs_list.append(image_path+str(file)[2:-1])
            
            #mask
                com = loadmat(masks_path[k] +'/COM.mat') 
              
                masks = com['MASK']
                masks_grouped_list.append(masks)
            
            for i in range(len(masks_grouped_list)): #len(img_list), start = len(masks_grouped_list) - len(img_list), end= len(img_list)
                masks = masks_grouped_list[i]
                for j in range(len(masks)):
                    masks_list.append(toPIL(masks[j])) #remove centerCrop
                
        return imgs_list, masks_list
    
    

