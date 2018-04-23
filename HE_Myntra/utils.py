import cv2
from scipy import ndimage
from skimage.transform import rotate
import random
import numpy as np
np.random.seed(42)

def randomZoomOut(image, zoom_ratio=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        im = image
        image = image.reshape(3, 75, 75)
        height, width = image[0].shape
        
        zoom = height*np.random.uniform(zoom_ratio[0], zoom_ratio[1])
        if int(zoom)<0:
            zoom = abs(int(zoom))
            return cv2.resize(im[zoom//2:-zoom//2, zoom//2:-zoom//2], (height, width))
             
        
        if zoom<0:
            zoom = 0
        return cv2.resize(cv2.copyMakeBorder(im, int(zoom//2), int(zoom//2), int(zoom//2), int(zoom//2),
                                   cv2.BORDER_REPLICATE), (height, width), interpolation=cv2.INTER_AREA)
    
    return image


def randomShift(image, shift_ratio=(0, 0.3), u=0.5):
    if np.random.random() < u:
        im = image
        image = image.reshape(3, 75, 75)
        height, width = image[0].shape
        
        lrr_shift = int(height*np.random.uniform(shift_ratio[0]/2, shift_ratio[1]/2))
        tbb_shift = int(height*np.random.uniform(shift_ratio[0]/2, shift_ratio[1]/2))
        lrl_shift = int(height*np.random.uniform(shift_ratio[0]/2, shift_ratio[1]/2))
        tbt_shift = int(height*np.random.uniform(shift_ratio[0]/2, shift_ratio[1]/2))
        
        return cv2.resize(im[lrl_shift:width-lrr_shift, tbt_shift:height-tbb_shift], (height, width))
    
    return image
        

def randomNoisy(im, u=0.5):
    if np.random.random() < u:
        noise_types = ["speckle"]
        noise_typ = random.choice(noise_types)
        
        image = im[0, :, :]
        if noise_typ == "s&p":
            row,col = image.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(im)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = []
            for coords_idx in range(im.shape[0]):
                coords.append([np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape])
            
            for coords_idx, coords_val in enumerate(coords):
                out[coords_idx, coords_val] = np.random.randint(0.95, 1.05)

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = []
            for coords_idx in range(im.shape[0]):
                coords.append([np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape])

            for coords_idx, coords_val in enumerate(coords):
                out[coords_idx, coords_val] = np.random.randint(-1.05, -0.95)

            return out
        elif noise_typ =="speckle":
            ch,row,col = im.shape
            gauss = np.random.randn(ch,row,col)/np.random.randint(3,6)
            gauss = gauss.reshape(ch,row,col)
            noisy = im + im * gauss
            return noisy
    
    return im
    

def randomErodeDilate(image, u=0.5):
    if np.random.random() < u:
        op_types = ["erode", "dilate"]
        op_typ = random.choice(op_types)
        
        kernel_size = np.random.randint(3, 4)
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        if op_typ=="erode":
            ims = []
            for im in image:
                ims.append(cv2.erode(im,kernel,iterations = 1))
            return np.stack(ims)
        elif op_typ=="dilate":
            ims = []
            for im in image:
                ims.append(cv2.dilate(im,kernel,iterations = 1))
            return np.stack(ims)

    return image


def randomDenoising(image, u=0.5):
    if np.random.random() < u:
        filter_ = np.random.randint(4,5)
        dst_0 = ndimage.median_filter(image[0],filter_)
        dst_1 = ndimage.median_filter(image[1],filter_)
        return np.stack((dst_0, dst_1))
    return image


def randomRotation(image, u=0.5):
    if np.random.random() < u:
        im = image
        image = image.reshape(3, 75, 75)
        rot = np.random.randint(-10,10)
        return rotate(im, rot, mode='symmetric')
    return image
