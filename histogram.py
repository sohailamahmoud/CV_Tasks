import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
 
def thresholding(img,threshold):
    arr = np.array(img)
    arr[arr < threshold] = 0
    arr[arr >= threshold] = 255
    return arr

def global_thresholding(img,threshold):
    Thresholded_img=thresholding(img,threshold)
    return Thresholded_img

def local_thresholding(img,th1,th2,th3,th4):
    ############splitting image###########
    h, w = img.shape
    half = w//2
    left_part = img[:, :half]
    # [:,:half] means all the rows and
    # all the columns upto index half
    right_part = img[:, half:] 
    # [:,half:] means all the rows and all
    # the columns from index half to the end
    # cv2.imshow is used for displaying the image
    half2 = h//2
    Top_left= left_part[:half2, :]
    Top_right=right_part [:half2, :]
    buttom_left=left_part[half2:, :]
    buttom_right=right_part[half2:, :]
    Thresholded_top_left=thresholding(Top_left,th1)
    Thresholded_top_right=thresholding(Top_right,th2)
    Thresholded_buttom_left=thresholding(buttom_left,th3)
    Thresholded_buttom_right=thresholding(buttom_right,th4)
    #########combining image##3#####
    im_v1 = cv2.vconcat([Thresholded_top_left,Thresholded_buttom_left ])
    im_v2 = cv2.vconcat([Thresholded_top_right,Thresholded_buttom_right ])
    new_img = cv2.hconcat([im_v1,im_v2 ])
    return new_img


def normalize(img):
  lmin= float(img.min())
  lmax=float(img.max())
  normalized = np.floor((img-lmin)/(lmax-lmin)*255.0)
# plt.imsave("images\\normalized.bmp",normalized, cmap='gray')
  cv2.imwrite("images\\normalized.bmp",normalized)



def equalization(image_matrix, bins=256):
    #image_matrix = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)
    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1
    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)
    return image_eq
