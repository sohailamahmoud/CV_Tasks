import numpy as np
import math
from scipy.interpolate import RectBivariateSpline
from skimage._shared.utils import _supported_float_type
from skimage.util import img_as_float
from skimage.filters import sobel



def ActiveContourSnake(image, snake, alpha=0.01, beta=0.1,w_line=0, w_edge=1, gamma=0.01,max_px_move=1.0,max_num_iter=2500, convergence=0.1,boundary_condition='periodic'):
    

    #handling negative number of iterations
    max_num_iter = int(max_num_iter)
    if max_num_iter <= 0:
        raise ValueError("max_num_iter should be >0.")
    convergence_order = 10
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    
    #handling invalid boundary condition option
    if boundary_condition not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n" +
                         "Should be one of: "+", ".join(valid_bcs)+'.')


    img = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    img = img.astype(float_dtype, copy=False)

    RGB = img.ndim == 3

    # Find edges using sobel:
    if w_edge != 0:
        if RGB:
            edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                    sobel(img[:, :, 2])]
        else:
            edge = [sobel(img)]
    else:
        edge = [0]

    # Superimpose intensity and edge images:
    if RGB:
        img = w_line*np.sum(img, axis=2) \
            + w_edge*sum(edge)
    else:
        img = w_line*img + w_edge*edge[0]

    # Interpolate for smoothness:
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    snake_xy = snake[:, ::-1]
    x = snake_xy[:, 0].astype(float_dtype)
    y = snake_xy[:, 1].astype(float_dtype)
    n = len(x)
    xsave = np.empty((convergence_order, n), dtype=float_dtype)
    ysave = np.empty((convergence_order, n), dtype=float_dtype)

    # Build snake shape matrix for Euler equation in double precision
    eye_n = np.eye(n, dtype=float)
    a = (np.roll(eye_n, -1, axis=0)
         + np.roll(eye_n, -1, axis=1)
         - 2 * eye_n)  # second order derivative, central difference
    b = (np.roll(eye_n, -2, axis=0)
         + np.roll(eye_n, -2, axis=1)
         - 4 * np.roll(eye_n, -1, axis=0)
         - 4 * np.roll(eye_n, -1, axis=1)
         + 6 * eye_n)  # fourth order derivative, central difference
    A = -alpha * a + beta * b

    # Impose boundary conditions different from periodic:
    sfixed = False
    if boundary_condition.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if boundary_condition.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if boundary_condition.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if boundary_condition.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True

    # implicit spline energy minimization and use float_dtype
    inv = np.linalg.inv(A + gamma * eye_n)
    inv = inv.astype(float_dtype, copy=False)


    # Explicit time stepping for image energy minimization:
    for i in range(max_num_iter):
        fx = intp(x, y, dx=1, grid=False).astype(float_dtype, copy=False)
        fy = intp(x, y, dy=1, grid=False).astype(float_dtype, copy=False)

        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
        x += dx
        y += dy

        # Convergence criteria needs to compare to a number of previous configurations since oscillations can occur.
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :])
                                 + np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break

    return np.stack([y, x], axis=1)


#chaincode of final snake function
def ChainCode(snake):
    print(len(snake[0]))
    chainCode=[]

    for i in range (len(snake)-1):
        x1=snake[i][0]
        y1=snake[i][1]
        x2=snake[i+1][0]
        y2=snake[i+1][1]
        if x1==x2:
            if y1> y2:
                for i in range (int(y2),int(y1)):
                    chainCode.append(2)
            else:
                for i in range (int(y1),int(y2)):
                    chainCode.append(6)
        elif y1 ==y2:
            if x1>x2:
                for i in range (int(x2),int(x1)):
                    chainCode.append(0)
            else:
                for i in range (int(x2),int(x1)):
                    chainCode.append(4)
        elif x1>x2 and y1>y2 :
            for i in range (int(x2),int(x1)):
                    chainCode.append(1)
        elif x1<x2 and y1>y2 :
            for i in range (int(y2),int(y1)):
                    chainCode.append(3)
        elif x1<x2 and y1<y2 :
            for i in range (int(x1),int(x2)):    
                chainCode.append(5)
        elif x1>x2 and y1<y2 :
            for i in range (int(y1),int(y2)):
                    chainCode.append(7)
    return chainCode 


#function of calculating area of contour
def areaOfContour(snake):
    a = 0
    x0,y0 = snake[0]
    for [x1,y1] in snake[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return abs(a)


#Calculate Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)



#function of calculating perimeter of contour
def perimeter(snake):
    #Calculate perimeter of the shape defined by an array of points
    n = len(snake)
    perim = 0
    for i in range(n):
        perim += distance(snake[i], snake[(i+1) % n])
    return perim
