import torch
import torch.nn as nn
from config_W_net import Config
import numpy as np

from torch import autograd


config = Config()




class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvModule, self).__init__()

        layers = [
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim), # Depthwise (3x3) through each channel
            nn.Conv2d(input_dim, output_dim, 1), # Pointwise (1x1) through all channels
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim), # Depthwise (3x3) through each channel
            nn.Conv2d(output_dim, output_dim, 1),# Pointwise (1x1) through all channels
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

def crop_img(tensor, target_tensor):
    target_size_x =target_tensor.size()[2]
    tensor_size_x =tensor.size()[2]
    delta_x = tensor_size_x - target_size_x
    
    if delta_x == 0 :
        x_start = 0
        
    elif delta_x%2 == 0 :
        delta_x = delta_x//2
        x_start = delta_x
        
    elif delta_x%2 == 1:
        delta_x = delta_x//2
        x_start = delta_x + 1
        
    
    target_size_y =target_tensor.size()[3]
    tensor_size_y =tensor.size()[3]
    delta_y = tensor_size_y - target_size_y
    
    if delta_y == 0 :
        y_start = 0
        
    elif delta_y%2 == 0 :
        delta_y = delta_y//2
        y_start = delta_y
        
    elif delta_y%2 == 1:
        delta_y = delta_y//2
        y_start = delta_y + 1
        
    return tensor[:, :, x_start : tensor_size_x - delta_x , y_start : tensor_size_y - delta_y]
    
    
class BaseNet(nn.Module): # 1 U-net
    def __init__(self, input_channels=1, encoder=config.encoderLayerSizes, decoder=config.decoderLayerSizes, output_channels=config.k):
        super(BaseNet, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.first_module = nn.Sequential(*layers)


        self.pool = nn.MaxPool2d(2, 2)
        
        self.enc_modules = nn.ModuleList([ConvModule(channels, 2*channels) for channels in encoder])


        decoder_out_sizes = [int(x/2) for x in decoder] # = [512, 256, 128]
        
        self.dec_transpose_layers = nn.ModuleList([nn.ConvTranspose2d(channels, channels//2, 2, stride=2) for channels in decoder]) # Stride of 2 makes it right size
        self.dec_modules = nn.ModuleList([ConvModule(2*channels_out, channels_out) for channels_out in decoder_out_sizes])
        
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 64, 2, stride=2)

        layers = [
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, output_channels, 1), # No padding on pointwise
            #nn.ReLU(),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.last_module = nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        
        for module in self.enc_modules:
         
            activations.append(module(self.pool(activations[-1]))) 
            
        x_ = activations.pop(-1)
       
        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            up_conv = upconv(x_)
            skip_connection = crop_img(skip_connection, up_conv)
            x_ = conv(torch.cat((skip_connection, up_conv), 1))



        
        last_skip_connection = activations[-1]
        last_up_conv = self.last_dec_transpose_layer(x_)
        last_skip_connection = crop_img(last_skip_connection, last_up_conv)
        segmentations = self.last_module(torch.cat((last_skip_connection,last_up_conv), 1))
        
        return segmentations


class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(input_channels=1, encoder=config.encoderLayerSizes,   #input_channels = 1 for 1 grey channel (instead of 3 rgb))
                                    decoder=config.decoderLayerSizes, output_channels=config.k)
        self.softmax = nn.Softmax2d()
        self.U_decoder = BaseNet(input_channels=config.k, encoder=config.encoderLayerSizes,
                                    decoder=config.decoderLayerSizes, output_channels=1)
        self.sigmoid = nn.Sigmoid()

    def forward_encoder(self, x):
        x9 = self.U_encoder(x)
        segmentations = self.softmax(x9)
        return segmentations

    def forward_decoder(self, segmentations):
        x18 = self.U_decoder(segmentations)
        reconstructions = self.sigmoid(x18)
        return reconstructions

    def forward(self, x):
        segmentations = self.forward_encoder(x)
        x_prime = self.forward_decoder(segmentations)
        return segmentations, x_prime
    

  