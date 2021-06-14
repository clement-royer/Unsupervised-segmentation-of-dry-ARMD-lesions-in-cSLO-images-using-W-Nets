#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
from datetime import datetime
import os
sys.path.append("/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/")




toTensor   = transforms.ToTensor()
config_path ="/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/config.py"

sys.path.append(os.path.dirname(os.path.expanduser(config_path)))
from config import Config
config = Config()

use_cuda = torch.cuda.is_available()


nChannel = 100
nConv = 2
num_superpixels = 10000
compactness = 100
lr = 0.1

maxIter = 1000
minLabels = 2
visualize = 1

path_results = "/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/kanezaki_segmentation/results/"
if not os.path.exists(path_results):
    os.makedirs(path_results)

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
def save_model(autoencoder, modelName):
    path = os.path.join("/media/thesard/DATA/stage_data_clement/ProjetClement/W-Net-Pytorch-master/kanezaki_segmentation/models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(autoencoder.state_dict(), path+'/model.config')

    
    
# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
            
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

modelName = str(datetime.now())
 
# load image
train_dataset = AutoencoderDataset_LSTM_n("train", 1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
nb_img = len(train_dataset)

data, output = next(iter(train_dataloader))




def forward_seg(data, path_results, patient, name_img):
    
    
    im = np.array(data.cpu()).astype('float64').squeeze(0)

    if use_cuda:
        data = data.cuda().unsqueeze(0)
    data = Variable(data)
    print(data.shape)
    
    # slic
    labels = segmentation.slic(im, compactness=compactness, n_segments=num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )
    
    # train
    model = MyNet( 1 )
    if use_cuda:
        model.cuda()
    model.train()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,1))
    
    


    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        
    
    
        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )
        if use_cuda:
            target = target.cuda()
        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    
        #print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
        print (batch_idx, '/', maxIter, ':', nLabels, loss.item())
    
        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break
        
  
    
    if not os.path.exists(path_results+ patient):
        os.makedirs(path_results+ patient) 
        
    
        
    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    
    cv2.imwrite(path_results+ patient + name_img, im_target_rgb) 
    

  
def loop_on_data(list_patients, display='True'):
 
    for k in range(len(list_patients)):
        img_path = list_patients[k]
        patient = img_path[-15:]
        list_imgs = get_files(img_path)
   
        
        print(patient)
     
        for i in range(len(list_imgs)-6,len(list_imgs)-1):#
            
            img = img_path+list_imgs[i]
            with open(img, 'rb') as f:
                img = Image.open(f).convert('L')
            img = toTensor(img).float().cuda()

                
            forward_seg(img, path_results, patient, str(i)+'.png')
         

  
  



path_gts = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_gt/"
path_imgs = "/media/thesard/DATA/stage_data_clement/ProjetClement/resized_processed_img/"


list_patients = get_imgs_path(path_imgs)

for h in range(6):
    list_patients.pop(0)

list_patients_gt = get_imgs_path(path_gts)

loop_on_data(list_patients)