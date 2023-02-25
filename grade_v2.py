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
import cv2
from collections import Counter, defaultdict
from statistics import mode
from skimage.transform import hough_line, hough_line_peaks, rotate
import math

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

def nonmax_supression(image, theta):
    
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
                maxval = max(image[i,j-1], image[i,j+1])
            elif (22.5 <= degree[i,j] <67.5):
                maxval = max(image[i-1,j-1], image[i+1,j+1])
            elif (67.5 <= degree[i,j]<112.5):
                maxval = max(image[i-1,j],image[i+1,j])
            else:
                maxval = max(image[i-1,j+1], image[i+1,j-1])

            if image[i,j] >= maxval:
                maxarr[i,j] = image[i,j]
    maxarr =  np.multiply(maxarr, 255.0 / maxarr.max())
    return maxarr

def threshold_hysteris(image, lower, upper):

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
    m,n = image.shape
    thresh_img = np.zeros((m,n))
    
    strong_i, strong_j = np.where(image >= upper)
    #print( strong_i, strong_j)
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
                            thresh_img[i,j] == strong_val
                else:
      
                    thresh_img[i,j] = 0
    return thresh_img


def find_houghlines(image):
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


    # xcos
    xcos_thetas = np.dot(x_nonzero.reshape((-1,1)), cos_thetas.reshape((1,-1)))
    # ysin
    ysin_thetas = np.dot(y_nonzero.reshape((-1,1)), sin_thetas.reshape((1,-1)))

    rho_list = np.round(xcos_thetas + ysin_thetas) + diag_len
    rho_list = rho_list.astype(np.uint16)
    for i in range(len(thetas)):
        rho, counts = np.unique(rho_list[:,i], return_counts=True)
        acc[rho, i] = counts

    return acc, rhos, thetas



## understand this function..
def hough_peaks(H, num_peaks, nhood_size):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        min_x, max_x, min_y, max_y = int(min_x), int(max_x), int(min_y), int(max_y)
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


    # a simple funciton used to plot a Hough Accumulator
def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure()
    fig.canvas.set_window_title(plot_title)
    	
    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.show()

def rotate_paper(img, theta, nrows, ncols):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. 
    '''
  
    image_test = img.copy()

    # convert theta from radian to degree
    angle = np.rad2deg(theta)
    #angle = theta
    print(f'angle: {angle}')
    print(f'nrows: {nrows}, ncols:{ncols}')
    #np.rad2deg(thetas[indicies[0][1]])
    if angle == 45 or angle ==-45:
        angle = -angle
    image_test = Image.fromarray(image_test*255, 'L')
    # rotate the image by the detected angle
    rotated = image_test.rotate(-angle)
    # plt.imshow(rotated, cmap="gray")
    # plt.show()
    x = (rotated.width-ncols)//2 +20
    y = (rotated.height-nrows)//2+20
    rotated = rotated.crop((x, y, x+ncols-20,y+nrows-50))

    print(f'rotated.shape: {rotated.width} {rotated.height}')
    return rotated

def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    thetalist = []
    linesx = []
    linesy = []
    m,n = img.shape
    angles_list = []
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        #print(x1,x2, y1, y2)
        thetax = np.abs(np.rad2deg(theta))
        if ((thetax >=0 and thetax <=0.7) or (thetax >=89.3 and thetax<=90)):
            thetalist.append(np.rad2deg(theta))

            if thetax ==0:
                linesx.append((x1,y1,x2,y2))
                cv2.line(img, (x1, y1), (x2, y2), (0, 255,0 ), 2)

            if thetax==90:
                linesy.append((x1,y1,x2,y2))
                cv2.line(img, (x1, y1), (x2, y2), (0, 255,0 ), 2)

    plt.imshow(img,cmap="gray")
    plt.show()
    # print(f' median theta : {np.median(thetalist)}, mode: {mode(thetalist)}')
    print(f'thetalist : {np.median(thetalist)}')
    linesx = sorted(linesx, key=lambda x: x[0])
    linesy = sorted(linesy, key=lambda x: x[1], reverse=True)
    return linesx, linesy, np.median(thetalist)

def extract_squares(linesx,linesy, image):
    contiguous_reg_x = defaultdict(lambda:[])
    contiguous_reg_y = []

    j=0
    for i in range(j, len(linesx)-1):
        if abs(linesx[i][0]-linesx[i+1][0]) <=35:
            contiguous_reg_x[linesx[j][0]].append(linesx[i+1])

        else:
            j+=1

    j=0
    contiguous_reg_y.append(linesy[1])
    for i in range(2, len(linesy)-1):
        if abs(linesy[i][1]-linesy[i+1][1]) >150:
            contiguous_reg_y.append(linesy[i])
            break
                
    contiguous_reg_x = dict(sorted(contiguous_reg_x.items(), key=lambda k: len(k[1]), reverse=True)[:3])
   
    boxes = defaultdict(lambda : [])
    for x in contiguous_reg_x:
        start, end = 0, len(contiguous_reg_x[x])-1
        boxes[x] = [(contiguous_reg_x[x][start][0],contiguous_reg_y[0][1],contiguous_reg_x[x][end][0],contiguous_reg_y[0][1]),
        (contiguous_reg_x[x][start][0],contiguous_reg_y[1][1],contiguous_reg_x[x][end][0],contiguous_reg_y[1][1]),
        (contiguous_reg_x[x][start][0],contiguous_reg_y[0][1],contiguous_reg_x[x][start][0],contiguous_reg_y[1][1]),
        (contiguous_reg_x[x][end][0],contiguous_reg_y[0][1],contiguous_reg_x[x][end][0],contiguous_reg_y[1][1])
        ]
        # cv2.line(image, (contiguous_reg_x[x][start][0], contiguous_reg_y[0][1]), (contiguous_reg_x[x][start][2],contiguous_reg_y[1][1]), (0, 255,0 ), 2)
        # cv2.line(image, (contiguous_reg_x[x][end][0],  contiguous_reg_y[0][1]), (contiguous_reg_x[x][end][2], contiguous_reg_y[1][1]), (0, 255,0 ), 2)
        # cv2.line(image, (contiguous_reg_x[x][start][0], contiguous_reg_y[0][1]), (contiguous_reg_x[x][end][0], contiguous_reg_y[0][3]), (0, 255,0 ), 2)
        # cv2.line(image, (contiguous_reg_x[x][start][0], contiguous_reg_y[1][1]), (contiguous_reg_x[x][end][0], contiguous_reg_y[1][3]), (0, 255,0 ), 2)

    for x in boxes.keys():
        #print(x)
        cv2.line(image, (boxes[x][0][0], boxes[x][0][1]), (boxes[x][0][2],boxes[x][0][3]), (0, 255,0 ), 2)
        cv2.line(image, (boxes[x][1][0], boxes[x][1][1]), (boxes[x][1][2],boxes[x][1][3]), (0, 255,0 ), 2)
        cv2.line(image, (boxes[x][2][0], boxes[x][2][1]), (boxes[x][2][2],boxes[x][3][3]), (0, 255,0 ), 2)
        cv2.line(image, (boxes[x][3][0], boxes[x][3][1]), (boxes[x][3][2],boxes[x][3][3]), (0, 255,0 ), 2)
    
    
    
    boxes = dict(sorted(boxes.items()))
    # divide width of boxes by 5 and height by 30
    answer_blocks = defaultdict(lambda: {})
    for box_idx, x in enumerate(boxes.keys()):
        boxwidth = abs(boxes[x][0][0] - boxes[x][0][2])
        boxheight = abs(boxes[x][2][1] - boxes[x][3][3])
        print(f'boxheight: {boxheight}')
        gap = boxwidth//5
        start = min(boxes[x][0][0], boxes[x][0][2])
        endx = max(boxes[x][0][0], boxes[x][0][2])
        xs = [start]
        for i in range(4):
            xs.append(start+gap*(i+1))
            xs.append(start+gap*(i+1))
        xs.append(endx)
        xs = sorted(xs)
        print(xs)
       
        ys = np.round(np.linspace(boxes[x][2][1], boxes[x][3][3], 30)).astype(int)
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
                    cv2.rectangle(image, (a[key][0], a[key][1]),(a[key][0]+a[key][2], a[key][1]+a[key][3]), (0, 255,0 ), 2)
                    crop = image[a[key][1]:a[key][1]+a[key][3], a[key][0]: a[key][0]+a[key][2]]
                    threshold = 140

                    # make all pixels < threshold black
                    binarized = 1.0 * (crop < threshold)
                    pixel_intensity = np.count_nonzero(binarized)
                    print(f'pixel_intensity: {pixel_intensity}')
                    if  pixel_intensity>=800.0:
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
            outer_region_crop = image[outer_region[ys[yi]][1]:outer_region[ys[yi]][1]+outer_region[ys[yi]][3], outer_region[ys[yi]][0]: outer_region[ys[yi]][0]+outer_region[ys[yi]][2]]
            threshold_outer = 150

            binarized_crop = 1.0 * (outer_region_crop < threshold_outer)
            pixel_intensity_crop = np.count_nonzero(binarized_crop)
            if  pixel_intensity_crop>=480:
                possible_answers_outside.append(yi)
           
        # for ans in possible_answers_outside:
        #     answer_blocks[box_idx][ans]+="x"
    plt.imshow(image)
    plt.show()
    print(answer_blocks)

    plt.imshow(image,cmap="gray")
    plt.show()



def get_edges(image):
    '''
    Generates edge map of an image by taking vertical and
    horizontal gradients and supressing non maximum values

    Parameters
    ----------
    image: np.array
        Gray scale image 
    '''
    image_blur = noise_reduction(image,3) # blur image
    gradient,theta = get_gradients(image_blur) # get image gradients
    print(f'theta: {np.rad2deg(theta)}')
    max_array = nonmax_supression(gradient, theta)
    edges = threshold_hysteris(max_array, 0,50)
   
    return edges




if __name__ == "__main__":
    # load an image
    image = Image.open(sys.argv[1])


    # filename = sys.argv[1].split(".jpg")[0]
    # for angle in range(-30, 35, 5):
    #         generate_rotated_sheets(image, filename, angle)


    image_gray = image.convert("L")
    image_gray = np.array(image_gray)
    
    #image_test = np.pad(image_gray, pad_width=(20,20), mode='constant',constant_values=0)
    
    #image_test = ndimage.rotate(image_test, -35)
    
    ## Find canny edges for image using my canny implementation
    # Discarded this implementation as canny edges were too noisy
    # for small values of kernel (3x3), and bad for values of kernel
    # from (5x5) onwards

    ## Finding canny edges using pillow's laplacian filter

    # convert np.array image to pillow image
    # find canny edges with pillow's laplacian filter
    image_test =  Image.fromarray(image_gray * 255 , 'L')

    canny_image = image_test.filter(ImageFilter.FIND_EDGES)
    imageio.imsave('canny_image.png', canny_image) # save canny image
    canny_image = np.array(canny_image) # convert canny to np.array for further processing
    acc, rhos, thetas = find_houghlines(canny_image) # 
    indicies, H = hough_peaks(acc, len(acc), nhood_size=9)
    linesx, linesy, median_tilt = hough_lines_draw(image_gray.copy(), indicies, rhos, thetas)

    # image_test = np.array(image_test)

    # # first angle corresponding to one of the page outlines
    # theta = thetas[indicies[0][1]]
    # corrected_image = rotate_paper(image_test.copy(),theta, 2200,1700)

    # canny_image_corrected = corrected_image.filter(ImageFilter.FIND_EDGES)
    # imageio.imsave('canny_image_corrected_image.png', canny_image_corrected)
    # canny_image_corrected = np.array(canny_image_corrected)
    # acc, rhos, thetas = find_houghlines(canny_image_corrected)
    # indicies, H = hough_peaks(acc, len(acc), nhood_size=11)
    # image_test = np.array(image_test)
    # corrected_image = np.array(corrected_image)

    # linesx, linesy, median_tilt = hough_lines_draw(corrected_image.copy(), indicies, rhos, thetas)

    # # plt.imshow(image_gray)
    # # plt.show()
    # # corrected_image = rotate_paper(image_gray.copy(),median_tilt, 2200,1700)

    # # corrected_image = np.array(corrected_image)
    extract_squares(linesx,linesy,image_gray.copy())
    
    ### handle outer writings, add proper threshold for non zero