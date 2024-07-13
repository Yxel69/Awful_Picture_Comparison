#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:57:07 2024

@author: shroomy
"""

import cv2
import numpy as np

# Function to compare two images
def compare_images(image_path1, image_path2):
    # Read the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the images
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Define the region of interest (ROI)
    x, y, w, h = roi
    roi_diff = diff[y:y+h, x:x+w]
    
    # Apply a different threshold to the ROI
    _, roi_thresh = cv2.threshold(roi_diff, threshold2, 255, cv2.THRESH_BINARY)
    
    # Insert the ROI threshold back into the thresholded image
    thresh[y:y+h, x:x+w] = roi_thresh
    
    # Use morphological operations to highlight the differences
    kernel = np.ones((1, 1), np.uint8)
    diff_highlighted = cv2.dilate(thresh, kernel, iterations=1)
    
    # Display the original images and the difference
    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.imshow("Difference", diff)
    cv2.imshow("Highlighted Differences", diff_highlighted)
    
    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Define the region of interest (x, y, width, height)
roi = (320, 120, 120, 120)  # Example values, adjust as needed

# Define the thresholds
threshold1 = 30  # Threshold for the entire image
threshold2 = 30  # Threshold for the ROI

# Paths to the images to compare
image_path1 = '<good_image_path>'
image_path2 = '<comparison_picture>'

# Compare the images
compare_images(image_path1, image_path2)
quit()



