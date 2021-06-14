from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os, shutil
import copy
import gc

from config import Config
import util
from model import WNet
from autoencoder_dataset import AutoencoderDataset, AutoencoderDataset_gan
from soft_n_cut_loss import soft_n_cut_loss, soft_n_cut_loss_modified,soft_n_cut_loss_modified2


def reconstruction_loss_(x, x_prime):
    criterion = nn.MSELoss()
    MSE_loss= criterion(x, x_prime)
    return MSE_loss


def train_1enc_1dec(inputs, autoencoder,optimizer):
    optimizer.zero_grad()
     

    segmentations, reconstructions = autoencoder(inputs)

   
    l_soft_n_cut = soft_n_cut_loss(inputs, segmentations)
    l_reconstruction = reconstruction_loss(inputs,reconstructions)

    loss = (l_reconstruction + l_soft_n_cut)
    loss.backward() 
    optimizer.step()
    return segmentations, reconstructions, loss
    
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

def train_2enc_1dec_2(inputs, autoencoder,optimizer1, optimizer2):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
   
    segmentations, reconstructions = autoencoder(inputs)
    
    l_soft_n_cut = soft_n_cut_loss(inputs, segmentations)
    l_reconstruction = reconstruction_loss_(inputs,reconstructions)

    


    l_soft_n_cut.backward(retain_graph=True)
    l_reconstruction.backward(retain_graph=True)          
       
    optimizer1.step()
    optimizer2.step()
    return segmentations, reconstructions,l_soft_n_cut, l_reconstruction
    
def main():
    print("PyTorch Version: ",torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")
        torch.cuda.empty_cache()

    config = Config()

    ###################################
    # Image loading and preprocessing #
    ###################################
    train_dataset = AutoencoderDataset_gan("train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)


    ###################################
    #          Model Setup            #
    ###################################

    autoencoder = WNet()
    if torch.cuda.is_available():   
        autoencoder = autoencoder.float().cuda()

    optimizer1 = torch.optim.Adam(autoencoder.U_encoder.parameters(),lr=0.0005, weight_decay=0.00001) 
    optimizer2 = torch.optim.Adam(autoencoder.parameters(),lr=0.001) 
    modelName = str(datetime.now())


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

            running_loss1 += l_reconstruction.item()
            running_loss2 += l_soft_n_cut.item()
            
            if i%10==0 :
                
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
        
        if epoch%50==0:
            f, axes = plt.subplots(2,1)
            axes[0].plot(train_loss_recon)
            axes[0].set_title("Reconstruction Loss")
            axes[1].plot(train_loss_softncut)
            axes[1].set_title("Soft n-cuts Loss")
            path_loss = os.path.join("./models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
            if not os.path.exists(path_loss):
                os.makedirs(path_loss)
            plt.savefig(path_loss+'/loss.png')
            plt.close()
    
   
        if config.saveModel:
            util.save_model(autoencoder, modelName)

    f, axes = plt.subplots(2,1)
    axes[0].plot(train_loss_recon)
    axes[0].set_title("Reconstruction Loss")
    axes[1].plot(train_loss_softncut)
    axes[1].set_title("Soft n-cuts Loss")
    
    path_loss = os.path.join("./models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    if not os.path.exists(path_loss):
        os.makedirs(path_loss)
    plt.savefig(path_loss+'/loss.png')
    plt.close()
    
if __name__ == "__main__":
    main()



