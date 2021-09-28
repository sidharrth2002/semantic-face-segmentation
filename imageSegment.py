# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'dataset/test'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################
def segmentImage(img):
   # Inputs
   # img: Input image, a 3D numpy array of row*col*3 in BGR format
   #
   # Output
   # outImg: segmentation image
   #
   #  ########################################################################
   #  ADD YOUR CODE BELOW THIS LINE

   # obtain HSV and LAB versions of the input image
   img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

   # denoise each version
   img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
   img_hsv_denoised = cv2.fastNlMeansDenoisingColored(img_hsv, None, 10, 10, 7, 21)
   img_lab_denoised = cv2.fastNlMeansDenoisingColored(img_lab, None, 10, 10, 7, 21)
   img_gray_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)

   # BACKGROUND MASK
   ret, mask_bg = cv2.threshold(img_gray_denoised, 145, 255, cv2.THRESH_BINARY)

   # HAIR MASK
   hair_lower = np.array([0, 0, 6])
   hair_upper = np.array([120, 120, 120])
   hair_mask = cv2.inRange(img_hsv_denoised, hair_lower, hair_upper)
   hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   hair_mask = cv2.erode(hair_mask, hair_kernel, iterations=1)
   hair_mask = cv2.dilate(hair_mask, hair_kernel, iterations=11)

   # SKIN MASK
   skin_lower = np.array([5, 70, 90])
   skin_upper = np.array([15, 155, 200])
   skin_mask = cv2.inRange(img_hsv, skin_lower, skin_upper)
   skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   skin_mask = cv2.erode(skin_mask, skin_kernel, iterations=1)
   skin_mask = cv2.dilate(skin_mask, skin_kernel, iterations=18)

   # MOUTH MASK
   mouth_lower = np.array([65, 147, 136])
   mouth_upper = np.array([130, 170, 152])
   mouth_region = img_lab_denoised.copy()
   cv2.rectangle(mouth_region, (0, 0), (mouth_region.shape[1], 370), (0, 0, 0), -1)
   cv2.rectangle(mouth_region, (0, 470), (mouth_region.shape[1], mouth_region.shape[0]), (0, 0, 0), -1)
   mouth_mask = cv2.inRange(mouth_region, mouth_lower, mouth_upper)
   mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   mouth_mask = cv2.morphologyEx(mouth_mask, cv2.MORPH_OPEN, mouth_kernel)

   # NOSE MASK
   nose_lower = np.array([8, 90, 135])
   nose_upper = np.array([12, 155, 185])
   nose_region = img_hsv.copy()
   # black out all regions but the nose
   # top half
   cv2.rectangle(nose_region, (0, 0), (nose_region.shape[1], 250), (0, 0, 0), -1)
   # bottom half
   cv2.rectangle(nose_region, (0, 360), (nose_region.shape[1], nose_region.shape[0]), (0, 0, 0), -1)
   # left of nose
   cv2.rectangle(nose_region, (0, 250), (140, 360), (0, 0, 0), -1)
   # right of nose
   cv2.rectangle(nose_region, (250, 250), (nose_region.shape[1], 360), (0, 0, 0), -1)
   nose_mask = cv2.inRange(nose_region, nose_lower, nose_upper)
   nose_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
   nose_mask = cv2.morphologyEx(nose_mask, cv2.MORPH_OPEN, nose_kernel, iterations=1)

   # EYE MASK
   # the hair mask will also extract the eyes, so need to subtract from it
   eye_region = hair_mask.copy()
   # black out the eyes on the hair mask
   cv2.rectangle(eye_region, (65, 230), (eye_region.shape[1] - 80, 300), (0, 0, 0), -1)
   # subtract the blacked out version from the hair mask
   eyes_only = hair_mask - eye_region
   eyes_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   eyes_only = cv2.erode(eyes_only, eyes_kernel, iterations=5)
   eyes_only = cv2.dilate(eyes_only, eyes_kernel, iterations=1)

   # final image starts with HSV colour space
   outImg = img_hsv.copy()
   # assign random colours initially to separate each part
   outImg = cv2.bitwise_or(outImg, (50, 50, 50), mask=hair_mask)
   outImg = cv2.bitwise_or(outImg, (255, 0, 0), mask=mask_bg)
   outImg = cv2.bitwise_or(outImg, (0, 255, 0), mask=skin_mask)
   outImg = cv2.bitwise_or(outImg, (0, 0, 255), mask=mouth_mask)
   outImg = cv2.bitwise_or(outImg, (0, 0, 0), mask=nose_mask)
   outImg = cv2.bitwise_or(outImg, (0, 100, 0), mask=eyes_only)
   # map from HSV to RGB
   outImg = cv2.cvtColor(outImg, cv2.COLOR_HSV2RGB)
   # map from RGB to GRAY
   outImg = cv2.cvtColor(outImg, cv2.COLOR_RGB2GRAY)
   # assign the appropriate colour intensities to each mask based on the requirements
   outImg[np.where(hair_mask)] = [1]
   outImg[np.where(mask_bg)] = [0]
   outImg[np.where(skin_mask)] = [5]
   outImg[np.where(mouth_mask)] = [2]
   outImg[np.where(nose_mask)] = [4]
   outImg[np.where(eyes_only)] = [3]

   return outImg

    # END OF YOUR CODE
    #########################################################################
