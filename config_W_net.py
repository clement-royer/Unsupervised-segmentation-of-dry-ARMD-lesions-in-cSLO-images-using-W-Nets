class Config():
    def __init__(self):
        self.input_size = 256
        self.batch_size =  8
 
        self.all_image = True
        

        self.Radius = 3

        self.k = 3 # Number of classes
        self.num_epochs = 300 
     
        self.data_dir = "D:/stage_data_clement/ProjetClement/DMLA-TimeLapse-Align-corrected-auto/"# Directory of images
        self.showdata = False 
        self.showTrainingOutput = True
        
        self.useInstanceNorm = False # Instance Normalization
        self.useBatchNorm = True # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.2 #try value between 0.2 and 0.4 

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256, 512]
        self.decoderLayerSizes = [1024, 512, 256]

        self.showSegmentationProgress = True
        self.segmentationProgressDir = './segmentation_results/'

        self.variationalTranslation = 0 # Pixels, 0 for off. 1 works fine

        self.saveModel = True
