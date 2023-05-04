from scipy.signal import find_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt


def OptimalThreshold(img: np.ndarray, threshold): 
    Back = img[np.where(img <= threshold)]
    obj = img[np.where(img > threshold)]
    BackMean = np.mean(Back)
    ObjMean = np.mean(obj)
    optimal_threshold = (BackMean + ObjMean)/2
    return optimal_threshold

def Optimal(image: np.ndarray):
    img = np.copy(image)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        pass
    x = img.shape[1] - 1
    y = img.shape[0] - 1
    BackMean =(int(img[0,0]) + int(img[0,x]) + int(img[y,0]) + int(img[y,x]))/4
    total=0
    count = 0
    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[0]):
            if not ((i == 0 and j == 0) or (i == x and j == 0) or (i == 0 and j == y) or (
                    i == x and j == y)):
                total += img[j, i]
                count += 1
    ObjMean = total/count
    threshold1= (BackMean + ObjMean)/2
    NewThreshold = OptimalThreshold(img,threshold1)
    OldThreshold=threshold1
    iter = 0
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = OptimalThreshold(img, OldThreshold)
        iter += 1

    img = np.copy(image)
    X=img.shape[0]
    Y=img.shape[1]
    for x in range(X):
        for y in range(Y):
            if img[x, y]<NewThreshold:
                img[x, y]=0
            else:
                img[x, y]=255
    return img




def otsu_threshold(image):
    # Compute the histogram of the image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Normalize the histogram so that it represents a probability distribution
    hist_norm = hist / float(image.size)
    
    # Compute the cumulative sum of the normalized histogram
    cum_sum = np.cumsum(hist_norm)
    
    # Compute the cumulative mean of the normalized histogram
    cum_mean = np.cumsum(hist_norm * np.arange(256))
    
    # Compute the global mean of the image
    global_mean = np.sum(hist_norm * np.arange(256))
    
    # Compute the between-class variance for each threshold value
    between_class_var = (global_mean * cum_sum - cum_mean)**2 / (cum_sum * (1 - cum_sum))
    
    # Find the threshold value that maximizes the between-class variance
    threshold = np.argmax(between_class_var)
    
    # Threshold the image using the computed threshold value
    binary_image = np.zeros_like(image)
    binary_image[image <= threshold] = 1
    
    return binary_image




def DoubleThreshold(Image, LowThreshold, HighThreshold, Weak, isRatio=True):
    
    # Get Threshold Values
    if isRatio:
        High = Image.max() * HighThreshold
        Low = Image.max() * LowThreshold
    else:
        High = HighThreshold
        Low = LowThreshold
    # Create Empty Array
    ThresholdedImage = np.zeros(Image.shape)

    Strong = 255
    # Find Position of Strong and Weak Pixels
    StrongRow, StrongCol = np.where(Image >= High)
    WeakRow, WeakCol = np.where((Image < High) & (Image >= Low))
    # Apply Thresholding
    ThresholdedImage[StrongRow, StrongCol] = Strong
    ThresholdedImage[WeakRow, WeakCol] = Weak

    return ThresholdedImage


def spectral(source: np.ndarray):
    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass
    # Dimensions of the image
    Y, X = src.shape
    HistValues = plt.hist(src.ravel(), 256)[0]
    PDF = HistValues / (Y* X)
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowThreshold in range(1, 254):
        for HighThreshold in range(LowThreshold + 1, 255):
            try:
                Back = np.arange(0, LowThreshold)
                Low = np.arange(LowThreshold, HighThreshold)
                High = np.arange(HighThreshold, 256)
                CDFL = np.sum(PDF[LowThreshold:HighThreshold])
                CDFH = np.sum(PDF[HighThreshold:256])
                BackMean = sum(Back * PDF[0:LowThreshold]) / CDF[LowThreshold]
                LowMean = sum(Low * PDF[LowThreshold:HighThreshold]) / CDFL
                HighMean = sum(High * PDF[HighThreshold:256]) / CDFH
                Variance = (CDF[LowThreshold] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                        CDFH * (HighMean - GMean) ** 2))
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowThreshold
                    OptimalHigh = HighThreshold
            except RuntimeWarning:
                pass
    return DoubleThreshold(src, OptimalLow, OptimalHigh, 128, False)



def local_thresholding(source: np.ndarray, Regions, ThresholdingFunction):
    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass
    YMax, XMax = src.shape
    Result = np.zeros((YMax, XMax))
    YStep = YMax // Regions
    XStep = XMax // Regions
    XRange = []
    YRange = []
    for i in range(0, Regions+1):
        XRange.append(XStep * i)
    for i in range(0, Regions+1):
        YRange.append(YStep * i)
    for x in range(0, Regions):
        for y in range(0, Regions):
            Result[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = ThresholdingFunction(src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]])
    return Result