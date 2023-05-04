import numpy as np
import cv2



# Compute the sum of squared differences (SSD) between the two images
def calculate_ssd(des1,des2):
    ssd = - (np.sqrt(np.sum((des1 - des2)**2)))
    return ssd



# Calculate the normalized cross-correlation (NCC) between the two sets of feature descriptors
def calculate_ncc(des1,des2):
    normlized_output1 = (des1 - np.mean(des1)) / (np.std(des1))
    normlized_output2 = (des2 - np.mean(des2)) / (np.std(des2))
    correlation_vector = np.multiply(normlized_output1, normlized_output2)
    ncc = float(np.mean(correlation_vector))
    return ncc



def feature_matching(des1,des2,method):


    keyPoints1 = des1.shape[0]
    keyPoints2 = des2.shape[0]

    #Store matching scores
    matched_features = []

    for kp1 in range(keyPoints1):
        # Initial variables (will be updated)
        distance = -np.inf
        y_index = -1
        for kp2 in range(keyPoints2):
            # Choose methode (ssd or normalized correlation)
            if method=="SSD":
               score = calculate_ssd(des1[kp1], des2[kp2])
            elif method =="NCC":
                score = calculate_ncc(des1[kp1], des2[kp2])


            if score > distance:
                distance = score
                y_index = kp2

        feature = cv2.DMatch()
        #The index of the feature in the first image
        feature.queryIdx = kp1
        # The index of the feature in the second image
        feature.trainIdx = y_index
        #The distance between the two features
        feature.distance = distance
        matched_features.append(feature)

    return matched_features