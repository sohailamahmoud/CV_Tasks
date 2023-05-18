
import random
import cv2
import numpy as np
from scipy import ndimage
from itertools import product
from numpy import zeros_like, ravel, sort, multiply, divide, int8
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros

#add noise

def Addnoise(img,btn1):
    if btn1=='Uniform':
       a=0 
       b=0.2
       row , col = img.shape
       Uni_Noise=np.zeros((row,col),dtype=np.float64)
       for i in range (row):
          for j in range (col):
            Uni_Noise[i][j]=np.random.uniform(a,b)
       result_image=img+Uni_Noise
       cv2.imwrite("images\\result_image.bmp",result_image)
       #return result_image
    elif btn1=='Gaussian':
        row , col= img.shape
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,size=(row,col))
        gauss = gauss.reshape(row,col)
        noisy = img + gauss
        cv2.imwrite("images\\noisy.bmp",noisy)
        #return noisy
    elif btn1=='salt & pepper':
        row , col = img.shape
        # for coloring them white
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1) # Pick a random y 
            x_coord=random.randint(0, col - 1) # Pick a random x
            img[y_coord][x_coord] = 255
        # coloring them black
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1) # Pick a random y 
            x_coord=random.randint(0, col - 1) # Pick a random x 
            # Color that pixel to black
            img[y_coord][x_coord] = 0
        return img


#Filtering

def MeanFilter(image, filter_size):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # creat an empty variable
    result = 0

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        result = result + image[j+y, i+x]
                output[j][i] = int(result / filter_size)
                result = 0

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                for y in range(-2, 3):
                    for x in range(-2, 3):
                        result = result + image[j+y, i+x]
                output[j][i] = int(result / filter_size)
                result = 0

    return output


def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g


def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i : i + k_size, j : j + k_size])
        image_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)

    # reshape and get the dst image
    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst

def median_filter(gray_img, mask=3):
    """
    :return: image with median filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    median_img = zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            # get mask according with mask
            kernel = ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])
            # calculate mask median
            median = sort(kernel)[int8(divide((multiply(mask, mask)), 2) + 1)]
            median_img[i, j] = median
    return median_img


    
    
    
   
   
    
#Edge detection   
    
    
# Sobel Edge Detection
def SobelXY(img):
    sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = sobelx + sobely
    return sobelxy


# Canny Edge Detection
def cannyDetection(img):
    edges = cv2.Canny(image=img, threshold1=100, threshold2=200) # Canny Edge Detection
    return edges


#prewitt
def prewitt(img):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty
    return img_prewitt


#Robert
def Robert(img):
    img_robert = img/255
    #Making the Robert's Crosses
    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                [ 0, 1, 0 ],
                                [ 0, 0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                [ 0, 0, 1 ],
                                [ 0,-1, 0 ]] )

    #Generate the edged image from G=(√Gx2+Gy2)
    vertical = ndimage.convolve( img_robert, roberts_cross_v )
    horizontal = ndimage.convolve( img_robert, roberts_cross_h )
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    return edged_img
    
