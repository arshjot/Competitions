import cv2
from scipy import ndimage
import random
import numpy as np
np.random.seed(42)

def randomZoomOut(im, zoom_ratio=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        height, width = im[0].shape
        
        zoom = height*np.random.uniform(zoom_ratio[0], zoom_ratio[1])
        if int(zoom)<0:
            zoom = abs(int(zoom))
            im_0 = cv2.resize(im[0, zoom//2:-zoom//2, zoom//2:-zoom//2], (height, width))
            im_1 = cv2.resize(im[1, zoom//2:-zoom//2, zoom//2:-zoom//2], (height, width))
#            im_2 = cv2.resize(im[2, zoom//2:-zoom//2, zoom//2:-zoom//2], (height, width))
#            im_3 = cv2.resize(im[3, zoom//2:-zoom//2, zoom//2:-zoom//2], (height, width))
            return np.stack((im_0,im_1))
        
        if zoom<0:
            zoom = 0
        im_0 = cv2.resize(cv2.copyMakeBorder(im[0],int(zoom//2),int(zoom//2),int(zoom//2),int(zoom//2),
                                   cv2.BORDER_REPLICATE), (height, width))
        im_1 = cv2.resize(cv2.copyMakeBorder(im[1],int(zoom//2),int(zoom//2),int(zoom//2),int(zoom//2),
                                   cv2.BORDER_REPLICATE), (height, width))
#        im_2 = cv2.resize(cv2.copyMakeBorder(im[2],int(zoom//2),int(zoom//2),int(zoom//2),int(zoom//2),
#                                   cv2.BORDER_REPLICATE), (height, width))
#        im_3 = cv2.resize(cv2.copyMakeBorder(im[3],int(zoom//2),int(zoom//2),int(zoom//2),int(zoom//2),
#                                   cv2.BORDER_REPLICATE), (height, width))
        return np.stack((im_0,im_1))
    
    return im


def randomShift(im, shift_ratio=(0, 0.3), u=0.5):
    if np.random.random() < u:
        height, width = im[0].shape
        
        lr_shift = int(height*np.random.uniform(shift_ratio[0], shift_ratio[1]))
        tb_shift = int(height*np.random.uniform(shift_ratio[0], shift_ratio[1]))
        
        im_0 = cv2.resize(im[0, 0:width-lr_shift, 0:height-tb_shift], (height, width))
        im_1 = cv2.resize(im[1, 0:width-lr_shift, 0:height-tb_shift], (height, width))
#        im_2 = cv2.resize(im[2, 0:width-lr_shift, 0:height-tb_shift], (height, width))
#            im_3 = cv2.resize(im[3, zoom//2:-zoom//2, zoom//2:-zoom//2], (height, width))
        return np.stack((im_0,im_1))
    
    return im
        

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
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
            out[:, coords] = np.random.randint(-20, 15)

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[:, coords] = -15
            return out
        elif noise_typ =="speckle":
            ch,row,col = im.shape
            gauss = np.random.randn(ch,row,col)/4
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
            erosion_0 = cv2.erode(image[0],kernel,iterations = 1)
            erosion_1 = cv2.erode(image[1],kernel,iterations = 1)
#            erosion_2 = cv2.erode(image[2],kernel,iterations = 1)
#            erosion_3 = cv2.erode(image[3],kernel,iterations = 1)
            return np.stack((erosion_0,erosion_1))
        elif op_typ=="dilate":
            dilate_0 = cv2.dilate(image[0],kernel,iterations = 1)
            dilate_1 = cv2.dilate(image[1],kernel,iterations = 1)
#            dilate_2 = cv2.dilate(image[2],kernel,iterations = 1)
#            dilate_3 = cv2.dilate(image[3],kernel,iterations = 1)
            return np.stack((dilate_0,dilate_1))

    return image


def randomDenoising(image, u=0.5):
    if np.random.random() < u:
        filter_ = np.random.randint(3,5)
        dst_0 = ndimage.median_filter(image[0],filter_)
        dst_1 = ndimage.median_filter(image[1],filter_)
        return np.stack((dst_0, dst_1))
    return image
