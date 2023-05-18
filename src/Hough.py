from copy import deepcopy
import cv2
import numpy as np
import cv2
import numpy as np
from math import sqrt, atan2, pi



def gaussian_smoothing(input_img):
                                
    gaussian_filter=np.array([[0.109,0.111,0.109],
                              [0.111,0.135,0.111],
                              [0.109,0.111,0.109]])
                                
    return cv2.filter2D(input_img,-1,gaussian_filter)



#defining the custom canny detector
def canny_edge_detection(input,l_threshold,u_threshold):
    
    input = input.astype('uint8')

    lower_threshold=l_threshold
    upper_threshold=u_threshold
    edges = cv2.Canny(input, lower_threshold, upper_threshold)
    cv2.imwrite('images\edge_Detected_Image.jpg',edges)
    return edges


#Hough circle function
def HoughCircles(input,circles,threshold): 
        th=threshold
        rows = input.shape[0] 
        cols = input.shape[1] 
    
        # initializing the angles to be computed 
        sinang = dict()
        cosang = dict()
        # initializing the angles  
        for angle in range(0,360): 
            sinang[angle] = np.sin(angle * np.pi/180) 
            cosang[angle] = np.cos(angle * np.pi/180) 
                

        length=int(rows/2)
        radius = [i for i in range(5,length)]
        for r in radius:
            #Initializing an empty 2D array with zeroes 
            acc_cells = np.full((rows,cols),fill_value=0,dtype=np.uint64)
            
            # Iterating through the original image 
            for x in range(rows): 
                for y in range(cols): 
                    if input[x][y] == 255:# edge 
                        # increment in the accumulator cells 
                        for angle in range(0,360): 
                            b = y - round(r * sinang[angle]) 
                            a = x - round(r * cosang[angle]) 
                            if a >= 0 and a < rows and b >= 0 and b < cols: 
                                acc_cells[a][b] += 1
                                
            acc_cell_max = np.amax(acc_cells)
            
            if(acc_cell_max > th):  

                print("Detecting the circles for radius: ",r)       
                
                # Initial threshold
                acc_cells[acc_cells < th] = 0  
                
                # find the circles for this radius 
                for i in range(rows): 
                    for j in range(cols): 
                        if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j] >= th):
                            avg_sum = np.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                            print("Intermediate avg_sum: ",avg_sum)
                            if(avg_sum >= 33):
                                print("For radius: ",r,"average: ",avg_sum,"\n")
                                circles.append((i,j,r))
                                acc_cells[i:i+5,j:j+7] = 0


#circle detection function
def Hough_circle_detection(path,threshold,l_threshold,u_threshold):
    circles = []
    orig_img = cv2.imread(path, cv2.IMREAD_COLOR)
    image_gray= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    input_img = deepcopy(image_gray)
    smoothed_img = gaussian_smoothing(input_img)
    edged_image=canny_edge_detection(smoothed_img,l_threshold=l_threshold,u_threshold=u_threshold)

    
    HoughCircles(input= edged_image,circles=circles,threshold=threshold)
    for vertex in circles:
        cv2.circle(orig_img,(vertex[1],vertex[0]),vertex[2],(0,255,0),1)
        cv2.rectangle(orig_img,(vertex[1]-2,vertex[0]-2),(vertex[1]-2,vertex[0]-2),(0,0,255),3)

    print(circles)
    cv2.imwrite('images\Circle_Detected_Image.jpg',orig_img)






#function of ellipse detection
def hough_ellipse(img, threshold=4, accuracy=1, min_size=4, max_size=None):
    if img.ndim != 2:
        raise ValueError('The input image must be 2D.')

    if not np.any(img):
        return np.zeros((0, 6))

    pixels = np.row_stack(np.nonzero(img))

    num_pixels = pixels.shape[1]
    acc = []
    results = []
    bin_size = accuracy * accuracy

    if max_size is None:
        if img.shape[0] < img.shape[1]:
            max_b_squared = np.round(0.5 * img.shape[0])
        else:
            max_b_squared = np.round(0.5 * img.shape[1])
        max_b_squared *= max_b_squared
    else:
        max_b_squared = max_size * max_size

    print("loop started")
    for p1 in range(num_pixels):
        p1x = pixels[1, p1]
        p1y = pixels[0, p1]

        for p2 in range(p1):
            p2x = pixels[1, p2]
            p2y = pixels[0, p2]

            dx = p1x - p2x
            dy = p1y - p2y
            a = 0.5 * sqrt(dx * dx + dy * dy)
            if a > 0.5 * min_size:
                xc = 0.5 * (p1x + p2x)
                yc = 0.5 * (p1y + p2y)

                for p3 in range(num_pixels):
                    p3x = pixels[1, p3]
                    p3y = pixels[0, p3]
                    dx = p3x - xc
                    dy = p3y - yc
                    d = sqrt(dx * dx + dy * dy)
                    if d > min_size:
                        dx = p3x - p1x
                        dy = p3y - p1y
                        cos_tau_squared = ((a*a + d*d - dx*dx - dy*dy) / (2 * a * d))
                        cos_tau_squared *= cos_tau_squared

                        k = a*a - d*d * cos_tau_squared
                        if k > 0 and cos_tau_squared < 1:
                            b_squared = a*a * d*d * (1 - cos_tau_squared) / k

                            if b_squared <= max_b_squared:
                                acc.append(b_squared)

                if len(acc) > 0:
                    bins = np.arange(0, np.max(acc) + bin_size, bin_size)
                    hist, bin_edges = np.histogram(acc, bins=bins)
                    hist_max = np.max(hist)
                    if hist_max > threshold:
                        orientation = atan2(p1x - p2x, p1y - p2y)
                        b = sqrt(bin_edges[hist.argmax()])
                        if orientation != 0:
                            orientation = pi - orientation
                            if orientation > pi:
                                orientation = orientation - pi / 2.
                                a, b = b, a
                        results.append((hist_max, yc, xc, a, b, orientation))
                    acc = []
                    
    print("Function Done")
    return np.array(results, dtype=[('accumulator', np.intp),
                                    ('yc', np.float64),
                                    ('xc', np.float64),
                                    ('a', np.float64),
                                    ('b', np.float64),
                                    ('orientation', np.float64)])



#hough line detection
def houghLines(edges, dTheta, threshold):
    imageShape = edges.shape
    imageDiameter = (imageShape[0]**2 + imageShape[1]**2)**0.5
    rhoRange = [i for i in range(int(imageDiameter)+1)]
    thetaRange = [dTheta*i for i in range(int(-np.pi/(2*dTheta)), int(np.pi/dTheta))]
    cosTheta = []
    sinTheta = []
    for theta in thetaRange:
        cosTheta.append(np.cos(theta))
        sinTheta.append(np.sin(theta))
    countMatrixSize = (len(rhoRange), len(thetaRange))
    countMatrix = np.zeros(countMatrixSize)

    eds = []
    for (x,y), value in np.ndenumerate(edges):
        if value > 0:
            eds.append((x,y))

    for thetaIndex in range(len(thetaRange)):
        theta = thetaRange[thetaIndex]
        cos = cosTheta[thetaIndex]
        sin = sinTheta[thetaIndex]
        for x, y in eds:
            targetRho = x*cos + y*sin
            closestRhoIndex = int(round(targetRho))
            countMatrix[closestRhoIndex, thetaIndex] += 1
            lines = []
    for (p,t), value in np.ndenumerate(countMatrix):
        if value > threshold:
            lines.append((p,thetaRange[t]))
    # print(lines)
    return lines


#drawing hough lines
def Hough_line_detection(path,l_threshold,u_threshold,threshold):
    image = cv2.imread(path)
    # Convert image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Use canny edge detection
    edges = cv2.Canny(gray,l_threshold,u_threshold,apertureSize=3)
    # np.pi/180 means one degree in rad
    lines =houghLines(edges, np.pi/180, threshold)
    lines =np.array(lines)

    for i in range(0, len(lines)):
        # print ((lines[i]))
        r = lines[i][0]
        theta = lines[i][1]
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
        # x0 stores the value rcos(theta)
        x0 = a*r
        # y0 stores the value rsin(theta)
        y0 = b*r
        # x1 stores value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
        # y1 stores value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
        # x2 stores value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
        # y2 stores value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 225), 2)

    cv2.imwrite('images/Detected_Line_Image.jpg', image)

