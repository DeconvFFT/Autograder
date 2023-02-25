from email.policy import default
from re import S
from PIL import Image
from PIL import ImageFilter
import sys
import random
import numpy as np
from scipy import fftpack
import imageio
import matplotlib.pyplot as plt

from scipy import signal, ndimage
import scipy
from collections import Counter, defaultdict

def noise_reduction(image,size=5):
    '''
    Applies gaussian blurring on an 
    image and facilitates in noise reduction.
    Uses fft to efficiently apply gaussian blurring

    Parameters
    ----------

    image: np.array()
        Gray scale image 
    
    size: int
        Size of the gaussian kernel
    
    Returns
    -------

    image_blur: np.array()
        A smoothed version of the original image
    '''
    #image = np.array(image)
    m,n = image.shape # get number of rows and number of columns of an image

    # Generate a gaussian kernel of shape mxn.
    # signal.gaussian generates matrix of shape mx1 
    # with 5 values in center distributed as gaussian and 
    # then generate matrix of shape nx1 in the same manner
    # Then, use np.outer to an outer join of those two matrices
    # and generate mxn kernel
    kernel = np.outer(signal.gaussian(m, size), signal.gaussian(n, size))

    # create fourier transform of image and kernel
    fft_image = fftpack.fft2(image) 
    fft_kernel = fftpack.fft2(fftpack.ifftshift(kernel))


    # multiply ffts of image and kernel
    # (A*B = ifft(fft(A) x fft(B)))
    fft_blur = fft_image*fft_kernel
    # get the blurred image by inverse fft
    image_blur = np.real(fftpack.ifft2(fft_blur))
    return image_blur
    


def get_gradients(image):

    '''
    Convolves a sobel filter on image and generates derviatives with respect to x and y.
    Uses the derivatives to get edge gradients and edge magnitutes.

    Parameters
    ----------

    image: np.array()
        Smoothed version of the original image
    
    Returns
    -------
        Gradient: np.array()
            Array containing edge gradient magnitudes for the image

        theta: np.array()
            Array containing edge gradient directions for the image
    '''
    # took idea inspiration from :https://dsp.stackexchange.com/questions/2830/can-edge-detection-be-done-in-the-frequency-domain
    # use sobel as an approximation of gaussian derviative
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/8

    dx = ndimage.convolve(image,sx)
    dy = ndimage.convolve(image, sy)
   
    # get image gradient magnitude
    gradient = np.hypot(dx,dy)
    # normalising gradient and converting it to 0-255 range
    gradient = np.multiply(gradient, 255.0 / gradient.max())
    
    # get the angle (direction) of gradient
    theta = np.arctan2(dy, dx)
    return gradient, theta

def nonmax_supression(gradients, theta):
    
    '''
    Compares the pixel values in 8 directions for one pixel and 
    sets the value of pixel to the maximum of those 8 values.

    Parameters
    ----------

    gradients: np.array()
        Edge gradients of an image
    
    theta: np.array()
        List of gradient angles 
    Returns
    -------
        maxarr: np.array()
            Array with pixels values with non-maximum values supressed
    '''

    # convert radians to degree
    degree = np.rad2deg(theta)
    # convert negative degree into positive degrees
    degree[degree <0] +=180
    m,n = degree.shape
    maxarr = np.zeros((m,n)) # array to hold maxvalues

    # angle comparision reference: https://web.stanford.edu/class/cs315b/assignment1.html
    maxval = -np.inf
    for i in range(1, m-1):
        for j in range(1, n-1):
            if (0<= degree[i,j] <22.5) or (157.5 <= degree[i,j] <= 180):
                maxval = max(gradients[i,j-1], gradients[i,j+1])
            elif (22.5 <= degree[i,j] <67.5):
                maxval = max(gradients[i-1,j-1], gradients[i+1,j+1])
            elif (67.5 <= degree[i,j]<112.5):
                maxval = max(gradients[i-1,j],gradients[i+1,j])
            else:
                maxval = max(gradients[i-1,j+1], gradients[i+1,j-1])

            if gradients[i,j] >= maxval:
                maxarr[i,j] = gradients[i,j] # supressing the non max values
    maxarr =  np.multiply(maxarr, 255.0 / maxarr.max()) 
    return maxarr

def threshold_hysteris(image, lower, upper):
    '''
    Creates strong edges and weak edges in thresholding step
    and tries to join weak edges with strong edges in hysteris step

    Parameters
    ----------

    image: np.array()
        Non max suppressed version of image
    
    lower: int
        lower threshold to remove weak edges
    
    upper: int
        high threshold to gather strong edges

    Returns
    -------
        thresh_img: np.array()
            Image with thinner edges 
    '''
    m,n = image.shape
    thresh_img = np.zeros((m,n))
    
    strong_i, strong_j = np.where(image >= upper)
    print( strong_i, strong_j)
    weak_i, weak_j = np.where((lower <= image) & (image <=upper))

    weak_val = 25
    strong_val = 255
    thresh_img[strong_i, strong_j] = strong_val
    thresh_img[weak_i, weak_j] = weak_val

    # join weak edges with strong edges
    # check in all 8 directions

    for i in range(1, m-1):
        for j in range(1, n-1):
            if thresh_img[i,j]==weak_val:
                if ((thresh_img[i,j-1]==strong_val) or (thresh_img[i,j+1]== strong_val) \
                    or (thresh_img[i-1,j-1]==strong_val) or (thresh_img[i+1,j+1]==strong_val) \
                        or (thresh_img[i-1,j]==strong_val) or (thresh_img[i+1,j]==strong_val) \
                            or (thresh_img[i-1,j+1]==strong_val) or (thresh_img[i+1,j-1]==strong_val)):
                            thresh_img[i,j] == strong_val # if pixel at i,j is weak pixel but has a strong pixel aroud it, make pixel strong
                else:
      
                    thresh_img[i,j] = 0 # if there are no strong pixels around the weak pixel, make it 0
    return thresh_img


def gather_houghlines(image):

    '''
    Gathers values of rho and theta for angles between -90 and 90. 
    Rho is xcostheta + ysintheta. 

    Parameters
    ----------

    image: np.array()
        Image after canny edge detection

    Returns
    -------
        acc: np.array()
            Collection of counts of rhos for one value of theta. 
    '''

    # referenced: https://www.mathworks.com/help/vision/ref/houghtransform.html to understand coordinate system of hough lines
    m,n  = image.shape
    # line with a parametric form: rho = xcos(theta) + ysin(theta)
    # theta ranges from -90 to 90
    thetas = np.deg2rad(np.arange(-90,90)) # collect thetas from -90 to 90
    diag_len = np.ceil(np.sqrt(m*m + n*n))
    rhos = np.arange(-diag_len, diag_len+1, 1)

    # collect rhos and thetas
    acc = np.zeros((len(rhos),len(thetas)))

    # collect all edge pixels
    y_nonzero, x_nonzero = np.nonzero(image)

    # cos and sin thetas
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)


    # array containing values of xcos theta
    xcos_thetas = np.dot(x_nonzero.reshape((-1,1)), cos_thetas.reshape((1,-1)))
    # array of values containing ysin theta
    ysin_thetas = np.dot(y_nonzero.reshape((-1,1)), sin_thetas.reshape((1,-1)))

    # rho = xcos theta + ysin theta
    rho_list = np.round(xcos_thetas + ysin_thetas) + diag_len
    rho_list = rho_list.astype(np.uint16)
    for i in range(len(thetas)):
        rho, counts = np.unique(rho_list[:,i], return_counts=True)
        acc[rho, i] = counts # gather counts of rhos for one theta

    return acc, rhos, thetas



def find_hough_peaks(acc, num_peaks,neighbours):
    
    # referenced slides and https://en.wikipedia.org/wiki/Hough_transform for inspiration on logic of hough lines
    # referenced algorithm from:  https://alyssaq.github.io/2014/understanding-hough-transform/
    '''
    Finds the maximum values and tries to supress the values in the neighbourhood. 

    Parameters
    ----------

    image: np.array()
        Image after canny edge detection

    Returns
    -------
        acc: np.array()
           Values of acc with peaks highlighted and with values in neighbourhood supressed
        
        peaks: list
            list of indices containing peaks of hough accumulator
    '''
    peaks = []
    acc_copy = np.copy(acc) # create a copy of acc so that we don't modify it in place

    
    for _ in range(num_peaks):
        idx = np.argmax(acc_copy) # find argmax in flattened array
        rho_theta_idx = np.unravel_index(idx, acc.shape) # remap to shape of H

        peaks.append(rho_theta_idx) # found peak at (x,y)
        theta_idx,rho_idx =  rho_theta_idx
        # check for rho neighbours
        h_neigh = neighbours/2
        if rho_idx - h_neigh < 0: # if neighbourhood size is such that it goes beyond value of rho (-ve), set minimum to rho
            min_rho = 0
        else: 
            min_rho = rho_idx - h_neigh # else, normal case. Just set minimum rho to rho-neighbourhood of rho

        if ((rho_idx + h_neigh + 1) > acc.shape[1]):  # here, neighbour hood size is big so it extends beyound limit of rho, 
            # so set max to size of rho array
            max_rho = acc.shape[1]
        else: max_rho = rho_idx + h_neigh + 1 # normal case, just set max to rho + neighbourhood of rho

        # check for theta neighbours
        v_neigh = neighbours/2

        if theta_idx - v_neigh < 0: # if neighbourhood size is such that it goes beyond value of theta (-ve), set minimum to theta
            min_theta = 0
        else: 
            min_theta = theta_idx - v_neigh # else, normal case. Just set minimum theta to theta-neighbourhood of theta

        if ((theta_idx + v_neigh + 1) > acc.shape[0]):  # here, neighbour hood size is big so it extends beyound limit of theta, 
            # so set max to size of theta array
            max_theta = acc.shape[0]
        else: max_theta = theta_idx + v_neigh + 1 # normal case, just set max to theta + neighbourhood of rthetaho



        min_rho, max_rho, min_theta, max_theta = int(min_rho), int(max_rho), int(min_theta), int(max_theta)
        for x in range(min_rho, max_rho):
            for y in range(min_theta, max_theta):
                acc_copy[y, x] = 0

                if (x == min_rho or x == (max_rho - 1)):
                    acc[y, x] = 255 # highlighting peaks in original acc
                if (y == min_theta or y == (max_theta - 1)):
                    acc[y, x] = 255 # highlighting peaks in original acc

    return peaks, acc


def gather_hvlines(indicies, rhos, thetas):
    '''
    Finds the horizontal and vertical lines. 

    Parameters
    ----------

    indices: np.array()
        indices containing peak of hough lines

    rhos: np.array()
        collection of rho values
    
    thetas : np.array()
        collection of theta valeus

    Returns
    -------
        acc: np.array()
           Values of acc with peaks highlighted and with values in neighbourhood supressed
    '''
    thetalist = []
    linesx = []
    linesy = []
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        thetalist.append(np.rad2deg(theta))

        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        thetax = np.abs(np.round(np.rad2deg(theta)))
        if (thetax ==0 or thetax==90):
            if thetax ==0:
                linesx.append((x1,y1,x2,y2))

            if thetax==90:
                linesy.append((x1,y1,x2,y2))
  
    linesx = sorted(linesx, key=lambda x: x[0])
    linesy = sorted(linesy, key=lambda x: x[1], reverse=True)

    return linesx, linesy


def extract_squares(linesx,linesy, image):
    '''
    Extracts form boxes and then uses those boxes to extract answer regions.
    Converts image to binary and uses threshold on answer region to find filled regions 

    Parameters
    ----------

    linesx: np.array()
        array containing vertical lines

    linesy: np.array()
        array containing horizontal lines
    
    image : np.array()
        original image

    Returns
    -------
        answer_blocks: dict
           dictionary containing answers for each row in each form box
    '''
    contiguous_reg_x = defaultdict(lambda:[])
    contiguous_reg_y = []

    j=0

    index_remove = []

    for i in range(1,len(linesx)):
        if abs(linesx[i-1][0]-linesx[i][0]) <=25:
            contiguous_reg_x[linesx[j][0]].append(linesx[i])
        else:

            j+=1
    

    contiguous_reg_y.append(linesy[1])
    for i in range(2, len(linesy)-1):
        if abs(linesy[i][1]-linesy[i+1][1]) >150:
            contiguous_reg_y.append(linesy[i])
            break
                
    contiguous_reg_x = dict(sorted(contiguous_reg_x.items(), key=lambda k: len(k[1]), reverse=True)[:3])
    k=0
    for x in contiguous_reg_x:
        index_remove = []
        
        for linei in  range(len(contiguous_reg_x[x])-1, 0,-1):
            
            dist = abs(contiguous_reg_x[x][linei][0] - contiguous_reg_x[x][linei-1][0])
            if dist>=15:

                index_remove.append(linei-1)
                
        contiguous_reg_x[x] = [contiguous_reg_x[x][i] for i in range(len(contiguous_reg_x[x])) if i not in index_remove]
        k+=1
    

    boxes = defaultdict(lambda : [])
    for x in contiguous_reg_x:
        start, end = 0, len(contiguous_reg_x[x])-1
        boxes[x] = [(contiguous_reg_x[x][start][0],contiguous_reg_y[0][1],contiguous_reg_x[x][end][0],contiguous_reg_y[0][1]),
        (contiguous_reg_x[x][start][0],contiguous_reg_y[1][1],contiguous_reg_x[x][end][0],contiguous_reg_y[1][1]),
        (contiguous_reg_x[x][start][0],contiguous_reg_y[0][1],contiguous_reg_x[x][start][0],contiguous_reg_y[1][1]),
        (contiguous_reg_x[x][end][0],contiguous_reg_y[0][1],contiguous_reg_x[x][end][0],contiguous_reg_y[1][1])]
        
    boxes = dict(sorted(boxes.items()))
    threshold = 200
    binarized = 1.0 * (image < threshold)

    # divide width of boxes by 5 and height by 30
    answer_blocks = defaultdict(lambda: {})
    for box_idx, x in enumerate(boxes.keys()):
        boxwidth = abs(boxes[x][0][0] - boxes[x][0][2])
        boxheight = abs(boxes[x][2][1] - boxes[x][3][3])
        gap = boxwidth//5

        start = min(boxes[x][0][0], boxes[x][0][2])
        endx = max(boxes[x][0][0], boxes[x][0][2])

        xs = [start]
        for i in range(4):
            xs.append(start+gap*(i+1))
            xs.append(start+gap*(i+1))
        xs.append(endx)
        xs = sorted(xs)

        ys = np.linspace(boxes[x][2][1], boxes[x][3][3], 30)
        ys = sorted(ys)
        blocks = defaultdict(lambda:[])
    
        for i in range(1, len(ys)):
            for j in range(1, len(xs),2):
                if ys[i-1] not in blocks:
                    blocks[ys[i-1]] = [{int(xs[j-1]): (int(xs[j-1]), int(ys[i-1]),
                    abs(int(xs[j])-int(xs[j-1])),abs(int(ys[i])-int(ys[i-1])))}]
                
                else:
                    blocks[ys[i-1]].append({int(xs[j-1]): (int(xs[j-1]), int(ys[i-1]),
                    abs(int(xs[j])-int(xs[j-1])),abs(int(ys[i])-int(ys[i-1])))})
     
        klist = list(blocks.keys())
        
        for idx, k in enumerate(klist):
            l = blocks[k]
            possible_answers = []

            for a in l:
                for key in a.keys():
                    crop = binarized[a[key][1]:a[key][1]+a[key][3], a[key][0]: a[key][0]+a[key][2]]
                    threshold = 150

                    pixel_intensity = np.count_nonzero(crop)
                   
                    if  pixel_intensity>=520.0:
                        possible_answers.append(key)
                    
            keys = [list(a.keys())[0] for a in l]
            answer = ""
            for ans in possible_answers:
                ind = keys.index(ans)
                answer+=chr(ind+65) 
            answer_blocks[box_idx][idx] = answer

        # check for outer regions
        outer_region = defaultdict(lambda:{})
        boxkeylist = list(boxes.keys())
        for j in range(1, len(ys)):
            if box_idx==0:
                outer_region[ys[j-1]] = (0,int(ys[j-1]),int(boxes[x][0][0]),abs(int(ys[j]-ys[j-1])))
            else:
                outer_region[ys[j-1]] = (int(boxes[boxkeylist[box_idx-1]][0][2]),int(ys[j-1]),abs(int(boxes[boxkeylist[box_idx-1]][0][2]-boxes[x][0][0])),abs(int(ys[j]-ys[j-1])))
        possible_answers_outside = []
        for yi in range(len(ys)-1):
            outer_region_crop = binarized[outer_region[ys[yi]][1]+10:outer_region[ys[yi]][1]+outer_region[ys[yi]][3], outer_region[ys[yi]][0]: outer_region[ys[yi]][0]+outer_region[ys[yi]][2]-5]
            pixel_intensity_crop = np.count_nonzero(outer_region_crop)
            if  pixel_intensity_crop>=350:
                possible_answers_outside.append(yi)
    
        for ans in possible_answers_outside:
            answer_blocks[box_idx][ans]+="x"

    return answer_blocks

def get_edges(image):

    '''
    Runs noise reduction to get smooth image, finds gradient magnitude and angle on the smooth image.
    Uses non-max supression to get strong edges and then uses hysterisis to obtain thin edges.

    Parameters
    ----------

    image: np.array()
        grayscale version of original image

    Returns
    -------
        edges: np.array
           image with edges highlighted
    '''

    image_blur = noise_reduction(image,3) # blur image
    gradient,theta = get_gradients(image_blur) # get image gradients
    max_array = nonmax_supression(gradient, theta)
    edges = threshold_hysteris(max_array, 0,30)

    
    return edges
if __name__ == "__main__":
    # load an image
    image = Image.open(sys.argv[1])
    image_gray = image.convert("L")
    image_gray = np.array(image_gray)
  
    canny_image = get_edges(image_gray)
    imageio.imsave('canny_image.png', canny_image)

    acc, rhos, thetas = gather_houghlines(canny_image)
    idxs, _ = find_hough_peaks(acc, len(acc),9)

    
    linesx, linesy = gather_hvlines(idxs, rhos, thetas)
    answers = extract_squares(linesx,linesy,image_gray.copy())

    output_file = sys.argv[2]
    done = False
    with open(output_file, 'w+') as f:
        i=1
        for key in answers.keys():
            if done:
                break
            for j in answers[key].keys():
                if i+j>85:
                    done  = True
                    break
                f.write(str(i+j) + " "+ answers[key][j]+"\n")
            i = i+len(answers[key].keys()) 