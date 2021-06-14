import cv2
import numpy as np
import math
import _utils
import argparse
import os.path
import torch
from sys import exit
import time
from tqdm import tqdm

# A simple convolution function that returns the filtered images.
def getFilterImages(filters, img):
    start_time = time.time()
    cuda = torch.device('cuda') 

    img = cv2.UMat(np.float32(img.cpu()))
    k = 0
    for filter in filters:
        if k==0:
            kern, params = filter
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            fimg = torch.tensor(fimg.get(),device=cuda)
            featureImages = fimg.unsqueeze(0)
        else :
            kern, params = filter
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            fimg = torch.tensor(fimg.get(),device=cuda)
            featureImages = torch.cat((featureImages,fimg.unsqueeze(0)),0)
        k+=1
    return featureImages




def filterSelection_gpu(featureImages, threshold, img, howManyFilterImages):
    start_time = time.time()
    cuda = torch.device('cuda') 
    id = 0
    height, width = img.shape
    k=0
    for featureImage in featureImages:
        featureImage = featureImage          
        thisEnergy = torch.tensor(0.0,dtype=torch.float64, device = cuda) 
        thisEnergy = torch.sum(torch.pow(featureImage,2))
       
        if k==0 :
            idEnergyList = torch.tensor((thisEnergy, id)).unsqueeze(0)
        else :
            idEnergyList = torch.cat((idEnergyList,torch.tensor((thisEnergy, id)).unsqueeze(0)), 0)
        id += 1
        k+=1
    E = torch.tensor(0.0, device = cuda)
    for E_i in idEnergyList:
        E += E_i[0]
    tempSum = torch.tensor(0.0, device = cuda)
    RSquared = torch.tensor(0.0, device = cuda)
    added = 0
    while ((RSquared < threshold) and (added < howManyFilterImages)):
        tempSum += sortedlist[added][0]
        RSquared = (tempSum/E)
        if added==0 :
            outputFeatureImages = (featureImages[int(sortedlist[added][1])]).unsqueeze(0)
        else:
            outputFeatureImages = torch.cat((outputFeatureImages,(featureImages[int(sortedlist[added][1])]).unsqueeze(0)),0)
        added += 1
    return outputFeatureImages

def build_filters(lambdas, ksize, gammaSigmaPsi):
    start_time = time.time()
    filters = []
    thetas = []
    thetas.extend([0, 45, 90, 135])
    thetasInRadians = [np.deg2rad(x) for x in thetas]

    for lamb in lambdas:
        for theta in thetasInRadians:
            params = {'ksize': (ksize, ksize), 'sigma': gammaSigmaPsi[1], 'theta': theta, 'lambd': lamb.item(),
                   'gamma':gammaSigmaPsi[0], 'psi': gammaSigmaPsi[2], 'ktype': cv2.CV_64F}
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append((kern, params))
    return filters
def getLambdaValues_gpu(img):
    start_time = time.time()
    cuda = torch.device('cuda')
    height, width = img.shape
    max = torch.tensor((width/4) * math.sqrt(2), device = cuda)
    min = torch.tensor(4 * math.sqrt(2), device = cuda)
    temp = min
    
    radialFrequencies = temp.unsqueeze(0)
    temp = temp * 2

    while(temp < max):
        radialFrequencies = torch.cat((radialFrequencies,temp.unsqueeze(0)),0)
        temp = temp * 2

    radialFrequencies = torch.cat((radialFrequencies,max.unsqueeze(0)),0)
    
    lambdaVals = (width/(radialFrequencies[0])).unsqueeze(0)
    for k in range(radialFrequencies.shape[0]):
        lambdaVals = torch.cat((lambdaVals,(width/(radialFrequencies[k])).unsqueeze(0)),0)
    
    return lambdaVals


def nonLinearTransducer_gpu(img, gaborImages, L, sigmaWeight, filters):
    start_time = time.time()
    cuda = torch.device('cuda') 
    
    alpha_ = torch.tensor(0.25, device = cuda)
    count = 0
    
    empty = True
    for gaborImage in gaborImages:
        gaborImage = gaborImage.double()

        avg = torch.mean(gaborImage)
        gaborImage = gaborImage - avg

        if int(cv2.__version__[0]) >= 3:
           
            
            gaborImage = cv2.UMat(np.float64(img.cpu()))
            gaborImage = cv2.normalize(gaborImage, gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            gaborImage = torch.tensor(gaborImage.get(),device=cuda)
        else:
            gaborImage = cv2.UMat(np.float64(img.cpu()))
            gaborImage = cv2.normalize(gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            gaborImage = torch.tensor(gaborImage.get(),device=cuda)

        copy = torch.abs(torch.tanh(gaborImage))

        copy, destroyImage = applyGaussian(copy, L, sigmaWeight, filters[count])
        if(not destroyImage):
            if empty :
                featureImages = copy.unsqueeze(0)
                empty = False
            else : 
                featureImages = torch.cat((featureImages, copy.unsqueeze(0)),0)
            
        count += 1
    return featureImages

def centralPixelTangentCalculation_bruteForce(img, copy, row, col, alpha, L):
    height, width = img.shape
    windowHeight, windowWidth, inita, initb = \
        _utils.getRanges_for_window_with_adjust(row, col, height, width, L)

    sum = 0.0
    for a in range(windowHeight + 1):
        for b in range(windowWidth + 1):
            truea = inita + a
            trueb = initb + b
            sum += math.fabs(math.tanh(alpha * (img[truea][trueb])))

    copy[row][col] = sum/pow(L, 2)

def applyGaussian(gaborImage, L, sigmaWeight, filter):
    cuda = torch.device('cuda')

    height, N_c = gaborImage.shape

    nparr = np.array(filter[0])
    u_0 = nparr.mean(axis=0)
    u_0 = u_0.mean(axis=0)

    destroyImage = False
    sig = 1
    if (u_0 < 0.000001):
        print("div by zero occured for calculation:")
        print("sigma = sigma_weight * (N_c/u_0), sigma will be set to zero")
        print("removing potential feature image!")
        destroyImage = True
    else:
        sig = sigmaWeight * (N_c / u_0)
        
    res = torch.tensor(cv2.GaussianBlur(cv2.UMat(np.float64(gaborImage.cpu())), (L, L), sig.item()).get(),device=cuda)

    return res, destroyImage

def removeFeatureImagesWithSmallVariance(featureImages, threshold):
    start_time = time.time()
    cuda = torch.device('cuda')
    empty = True
    for image in featureImages:
        if(torch.var(image) > threshold):
            if empty :
                toReturn = image.unsqueeze(0)
                empty = False
            else :
                toReturn= torch.cat((toReturn,image.unsqueeze(0)),0) 
    return toReturn

def runGabor(args):

    infile = args.infile
    if(not os.path.isfile(infile)):
        print(infile, " is not a file!")
        exit(0)

    outfile = args.outfile
    printlocation = os.path.dirname(os.path.abspath(outfile))
    _utils.deleteExistingSubResults(printlocation)

    M_transducerWindowSize = args.M
    if((M_transducerWindowSize % 2) == 0):
        print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1

    k_clusters = args.k
    k_gaborSize = args.gk

    spatialWeight = args.spw
    gammaSigmaPsi = []
    gammaSigmaPsi.append(args.gamma)
    gammaSigmaPsi.append(args.sigma.item())
    gammaSigmaPsi.append(args.psi)
    variance_Threshold = args.vt
    howManyFeatureImages = args.fi
    R_threshold = args.R
    sigmaWeight = args.siw
    greyOutput = args.c
    printIntermediateResults = args.i

    if int(cv2.__version__[0]) >= 3:
        img = cv2.imread(infile, 0)
    else:
        img = cv2.imread(infile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    lambdas = getLambdaValues(img)
    filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)

    print("Gabor kernels created, getting filtered images")
    filteredImages = getFilterImages(filters, img)
    filteredImages = filterSelection(filteredImages, R_threshold, img, howManyFeatureImages)
    if(printIntermediateResults):
        _utils.printFeatureImages(filteredImages, "filter", printlocation)

    print("Applying nonlinear transduction with Gaussian smoothing")
    featureImages = nonLinearTransducer(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
    featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)

    if (printIntermediateResults):
        _utils.printFeatureImages(featureImages, "feature", printlocation)

    featureVectors = _utils.constructFeatureVectors(featureImages, img)
    featureVectors = _utils.normalizeData(featureVectors, False, spatialWeight=spatialWeight)

    print("Clustering...")
    labels = _utils.clusterFeatureVectors(featureVectors, k_clusters)
    _utils.printClassifiedImage(labels, k_clusters, img, outfile, greyOutput)



infile = "./testImages/2.png"
outfile = "./out.png"
k = 2
gk = 13
sigma = 7 

def runGabor_without_parse_arg(inFile, outFile, k, gk, M,
                           spw, gamma, sigma, psi, vt, fi, R, siw, c, i):

    infile = inFile
    if(not os.path.isfile(infile)):
        print(infile, " is not a file!")
        exit(0)

    outfile = outFile
    printlocation = os.path.dirname(os.path.abspath(outfile))
    _utils.deleteExistingSubResults(printlocation)

    M_transducerWindowSize = M
    if((M_transducerWindowSize % 2) == 0):
        print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1

    k_clusters = k
    k_gaborSize = gk

    spatialWeight = spw
    gammaSigmaPsi = []
    gammaSigmaPsi.append(gamma)
    gammaSigmaPsi.append(sigma)
    gammaSigmaPsi.append(psi)
    variance_Threshold = vt
    howManyFeatureImages = fi
    R_threshold = R
    sigmaWeight = siw
    greyOutput = c
    printIntermediateResults = i

    if int(cv2.__version__[0]) >= 3:
        img = cv2.imread(infile, 0)
    else:
        img = cv2.imread(infile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    lambdas = getLambdaValues(img)
    filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)

    print("Gabor kernels created, getting filtered images")
    filteredImages = getFilterImages(filters, img)
    filteredImages = filterSelection(filteredImages, R_threshold, img, howManyFeatureImages)
    if(printIntermediateResults):
        _utils.printFeatureImages(filteredImages, "filter", printlocation)

    print("Applying nonlinear transduction with Gaussian smoothing")
    featureImages = nonLinearTransducer(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
    featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)

    if (printIntermediateResults):
        _utils.printFeatureImages(featureImages, "feature", printlocation)

    featureVectors = _utils.constructFeatureVectors(featureImages, img)
    featureVectors = _utils.normalizeData(featureVectors, False, spatialWeight=spatialWeight)

    print("Clustering...")
    labels = _utils.clusterFeatureVectors(featureVectors, k_clusters)
    _utils.printClassifiedImage(labels, k_clusters, img, outfile, greyOutput)
    
def runGabor_without_parse_arg_gpu(inFile, outFile, k, gk, M,
                           spw, gamma, sigma, psi, vt, fi, R, siw, c, i):

    
    outfile = outFile
    printlocation = os.path.dirname(os.path.abspath(outfile))

    M_transducerWindowSize = M
    if((M_transducerWindowSize % 2) == 0):
        print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1

    k_clusters = k
    k_gaborSize = gk

    gammaSigmaPsi = []
    gammaSigmaPsi.append(gamma.item())
    gammaSigmaPsi.append(sigma.item())
    gammaSigmaPsi.append(psi.item())
    variance_Threshold = vt
    howManyFeatureImages = fi
    R_threshold = R
    sigmaWeight = siw
    greyOutput = c
    printIntermediateResults = False
    
    with torch.cuda.device(0):
    
        img = torch.tensor(inFile).cuda()
                      

            
        lambdas = getLambdaValues_gpu(img)
        filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)
    
        filteredImages = getFilterImages(filters, img)
        filteredImages = filterSelection_gpu(filteredImages, R_threshold, img, howManyFeatureImages)
        
        if(printIntermediateResults):
            _utils.printFeatureImages(filteredImages, "filter", printlocation)
    
        featureImages = nonLinearTransducer_gpu(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
        featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)
    
        if (printIntermediateResults):
            _utils.printFeatureImages(featureImages, "feature", printlocation)
            
        featureVectors = _utils.constructNormalizedFeatureVectors(featureImages, img, setMeanToZero=False, spatialWeight=1)   
            
 
        labels, u, u0 = _utils.clusterFeatureVectors_gpu(featureVectors, k_clusters)
        
        
        segmentation = _utils.printClassifiedImage(labels, k_clusters, img, outfile, greyOutput)
        
    return segmentation, u, u0

def main():

    # initialize
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("-infile", required=True)
    parser.add_argument("-outfile", required=True)

    parser.add_argument('-k', help='Number of clusters', type=_utils.check_positive_int, required=True)
    parser.add_argument('-gk', help='Size of the gabor kernel', type=_utils.check_positive_int, required=True)
    parser.add_argument('-M', help='Size of the gaussian window', type=_utils.check_positive_int, required=True)

    # Optional arguments
    parser.add_argument('-spw', help='Spatial weight of the row and columns for clustering, DEFAULT = 1', nargs='?', const=1,
                        type=_utils.check_positive_float, default=1, required=False)
    parser.add_argument('-gamma', help='Spatial aspect ratio, DEFAULT = 1', nargs='?', const=1, default=1,
                        type=_utils.check_positive_float, required=False)
    parser.add_argument('-sigma', help='Spread of the filter, DEFAULT = 1', nargs='?', const=1, default=1,
                        type=_utils.check_positive_float, required=False)
    parser.add_argument('-psi', help='Offset phase, DEFAULT = 0', nargs='?', const=0, default=0,
                        type=_utils.check_positive_float, required=False)
    parser.add_argument('-vt', help='Variance Threshold, DEFAULT = 0.0001', nargs='?', const=0.0001, default=0.0001,
                        type=_utils.check_positive_float, required=False)
    parser.add_argument('-fi', help='Maximum number of feature images wanted, DEFAULT = 100', nargs='?', const=100, default=100,
                        type=_utils.check_positive_int, required=False)
    parser.add_argument('-R', help='Energy R threshold, DEFAULT = 0.95', nargs='?', const=0.95, default=0.95,
                        type=_utils.check_positive_float, required=False)
    parser.add_argument('-siw', help='Sigma weight for gaussian smoothing, DEFAULT = 0.5', nargs='?', const=0.5, default=0.5,
                        type=float, required=False)
    parser.add_argument('-c', help='Output grey? True/False, DEFAULT = False', nargs='?', const=False, default=False,
                        type=bool, required=False)
    parser.add_argument('-i', help='Print intermediate results (filtered/feature images)? True/False, DEFAULT = False', nargs='?', const=False, default=False,
                        type=bool, required=False)

    args = parser.parse_args()
    runGabor(args)

if __name__ == "__main__":
    cuda = torch.device('cuda') 
    main()
    
    
