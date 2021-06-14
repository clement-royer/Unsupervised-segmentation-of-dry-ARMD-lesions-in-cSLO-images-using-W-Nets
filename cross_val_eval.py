#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import gc

from config import Config
import util
from model import WNet

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

# for k in range(8,nb_cross):
#     #images
#     res = get_imgs_path(path4)
#     #gt
#     res_gt =get_imgs_path(path_gt)

   
#     test_path = []
#     test_path_gt = []
    
#     for i in range(6):
#         random_index = random.randint(0,len(res)-1)
#         test_path.append(res.pop(random_index))
#         test_path_gt.append(res_gt.pop(random_index))
    
#     train_path = res
#     train_path_gt = res_gt
    
    
#     print(len(train_path), len(test_path))
    
#     #copy train
#     for l in range (len(train_path)):
#         src = train_path[l]
#         dest = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval/" +str(k)+"/train/"+src[-15:]
#         destination = shutil.copytree(src, dest) 
        
#         src_gt = train_path_gt[l]
#         dest_gt = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval_gt/" +str(k)+"/train/"+src[-15:]
#         destination = shutil.copytree(src_gt, dest_gt) 
#     #copy test
#     for m in range (len(test_path)):
#         src = test_path[m]
#         dest = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval/" +str(k)+"/test/"+src[-15:]
#         destination = shutil.copytree(src, dest) 
        
#         src_gt = test_path_gt[m]
#         dest_gt = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval_gt/" +str(k)+"/test/"+src[-15:]
#         destination = shutil.copytree(src_gt, dest_gt) 
 
def save_model(autoencoder, modelName, index_crossval):
    path = os.path.join("./crossval_model/"+str(index_crossval)+"/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(autoencoder, path+'/model.config')
    # with open(path+".config", "a+") as f:
    #     f.write(str(config))
    #     f.close()
   
def reconstruction_loss_(x, x_prime):
    criterion = nn.MSELoss()
    MSE_loss= criterion(x, x_prime)
    return MSE_loss

def train_2enc_1dec(inputs, autoencoder,optimizer1, optimizer2):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
     
   # First optimization step (only encoder)
    segmentations, reconstructions = autoencoder(inputs)
    l_soft_n_cut = soft_n_cut_loss_modified2(inputs, segmentations)
    l_soft_n_cut.backward()
    optimizer1.step()
    
    optimizer2.zero_grad()
    segmentations, reconstructions = autoencoder(inputs)
    l_reconstruction = reconstruction_loss_(inputs,reconstructions)
    
    l_reconstruction.backward() 
    optimizer2.step()
    return segmentations, reconstructions,l_soft_n_cut, l_reconstruction

    
def main():
    print("PyTorch Version: ",torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")
        torch.cuda.empty_cache()

    config = Config()
    
    index_crossval = 8
    #jusqu'a 7 ok 
    
    path_img =  "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval/"+str(index_crossval)+"/train/"
    path_gt = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval_gt/"+str(index_crossval)+"/train/"
    
    path_img_test =  "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval/"+str(index_crossval)+"/test/"
    path_gt_test ="/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/cross_eval_gt/"+str(index_crossval)+"/test/"

    
    train_dataset = AutoencoderDataset_crossval(path_img,path_gt)
    val_dataset   = AutoencoderDataset_crossval(path_img_test,path_gt_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=1, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config.val_batch_size, num_workers=1, shuffle=True)

    util.clear_progress_dir()

    ###################################
    #          Model Setup            #
    ###################################

    autoencoder = WNet()
    if torch.cuda.is_available():   
        autoencoder = autoencoder.float().cuda()
        

    optimizer1 = torch.optim.Adam(autoencoder.U_encoder.parameters(),lr=0.0005, weight_decay=0.00001) 
    optimizer2 = torch.optim.Adam(autoencoder.parameters(),lr=0.001) 
    

    
    util.enumerate_params([autoencoder])

    # Use the current time to save the model at end of each epoch
    modelName = str(datetime.now())


    
    WEIGHT_CLIP = 0.01


    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.train() #Put model in training mode
    
    train_loss_softncut = []
    train_loss_recon = []
    
    
    for epoch in range(config.num_epochs):
        running_loss1 = 0.0
        running_loss2 = 0.0
        for i, [inputs, outputs] in enumerate(train_dataloader, 0):
            
            progress_images, progress_expected = next(iter(val_dataloader))

            if config.showdata:
                plt.imshow(inputs[0].squeeze(0).squeeze(0), cmap = 'Greys_r')
                plt.show()
                plt.imshow(outputs[0].squeeze(0).squeeze(0), cmap = 'Greys_r')
                plt.show()
                plt.close()

            if torch.cuda.is_available():
                inputs  = inputs.float().cuda()
                outputs = outputs.float().cuda()


            segmentations, reconstructions, l_soft_n_cut, l_reconstruction = train_2enc_1dec(inputs, autoencoder, optimizer1,optimizer2)


            # print statistics
            running_loss1 += l_reconstruction.item()
            running_loss2 += l_soft_n_cut.item()
            
            if i%50==0 :
                
                f, axes = plt.subplots(4, config.batch_size, figsize=(48,48))
                for l in range(config.batch_size):
                
                    axes[0, l].imshow((torch.argmax(segmentations[l], axis=0).float() / config.k).squeeze(0).detach().cpu())
                    axes[1, l].imshow(reconstructions[l].squeeze(0).detach().cpu())
                    axes[2, l].imshow(outputs[l].squeeze(0).detach().cpu())
                    axes[3, l].imshow(inputs[l].squeeze(0).detach().cpu())
                    plt.show()
                    plt.close()
            
            
            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss = running_loss1 / len(train_dataloader.dataset)
        epoch_loss2 = running_loss2 / len(train_dataloader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
        
        train_loss_recon.append(epoch_loss)
        train_loss_softncut.append(epoch_loss2)
            
   
        if config.saveModel:
            save_model(autoencoder, modelName, index_crossval)

    f, axes = plt.subplots(2,1)
    axes[0].plot(train_loss_recon)
    axes[0].set_title("Reconstruction Loss")
    axes[1].plot(train_loss_softncut)
    axes[1].set_title("Soft n-cuts Loss")
   
    path_loss = os.path.join("./crossval_model/"+str(index_crossval)+"/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    if not os.path.exists(path_loss):
        os.makedirs(path_loss)
    plt.savefig(path_loss+'/loss.png')
    plt.close()
    
if __name__ == "__main__":
     main()
