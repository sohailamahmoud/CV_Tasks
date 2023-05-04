import cv2
import numpy as np
import time

def harrisCorner(img):
   # Define the k value
    k = 0.04
    # Load the image in grayscale
    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # img = np.float32(img)

    img = img.astype(np.float32) / 255.0



    # Start the timer
    start_time = time.time()

    # Compute the x and y derivatives using the Sobel operator
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the elements of the structure tensor M
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)

    # Compute the sums of the structure tensor elements over a local window
    window_size = 3
    Sx2 = cv2.boxFilter(Ix2, -1, (window_size, window_size))
    Sy2 = cv2.boxFilter(Iy2, -1, (window_size, window_size))
    Sxy = cv2.boxFilter(Ixy, -1, (window_size, window_size)) 
    

    # Compute the Harris response for each pixel
    k = 0.04
    R = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            M = np.array([[Sx2[y, x], Sxy[y, x]], [Sxy[y, x], Sy2[y, x]]])
            eigenvals, _ = np.linalg.eig(M)
            R[y, x] = np.prod(eigenvals) - k * np.sum(eigenvals)

    # Threshold the Harris response to obtain corner candidates
    threshold = 0.01 * np.max(R)
    corners = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if R[y, x] > threshold:
                corners.append((x, y))

    
    # Apply non-maximum suppression to get the final corner locations
    window_size = 5
    corners_final = []
    for corner in corners:
        x, y = corner
        corner_val = R[y, x]
        window = R[max(y-window_size, 0):min(y+window_size+1, img.shape[0]),
                max(x-window_size, 0):min(x+window_size+1, img.shape[1])]
        if np.max(window) == corner_val:
            corners_final.append(corner)

    # Draw the corners on the original image and show the result
    img_with_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for corner in corners_final:
        cv2.circle(img_with_corners, corner, 5, (0, 0, 255), 2)
    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time


    return img_with_corners, execution_time