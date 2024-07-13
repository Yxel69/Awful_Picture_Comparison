import cv2
import numpy as np

def align_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Use ORB to detect and compute keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # Use homography to warp image2 to image1's perspective
    height, width = image1.shape[:2]
    aligned_image2 = cv2.warpPerspective(image2, h, (width, height))
    
    return aligned_image2

def adjust_exposure(image1, image2):
    # Convert images to LAB color space
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l1, a1, b1 = cv2.split(lab1)
    l2, a2, b2 = cv2.split(lab2)
    
    # Calculate mean and standard deviation of the L channel
    l1_mean, l1_stddev = cv2.meanStdDev(l1)
    l2_mean, l2_stddev = cv2.meanStdDev(l2)
    
    # Normalize the L channel of image2 to match the mean and stddev of image1
    l2 = ((l2 - l2_mean[0][0]) / l2_stddev[0][0]) * l1_stddev[0][0] + l1_mean[0][0]
    l2 = np.clip(l2, 0, 255).astype(np.uint8)
    
    # Merge channels back and convert to BGR color space
    adjusted_lab2 = cv2.merge((l2, a2, b2))
    adjusted_image2 = cv2.cvtColor(adjusted_lab2, cv2.COLOR_LAB2BGR)
    
    return adjusted_image2

# Function to compare two images
def compare_images(image_path1, image_path2, threshold1, threshold2,roithreshold):
    # Read the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Align the images
    aligned_image2 = align_images(image1, image2)
    
    # Adjust exposure
    adjusted_image2 = adjust_exposure(image1, aligned_image2)
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(adjusted_image2, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the images
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply a binary threshold to the entire image
    _, thresh1 = cv2.threshold(diff, threshold1, 255, cv2.THRESH_BINARY)
    
    # Apply a different threshold to the entire image (if needed)
    _, thresh2 = cv2.threshold(diff, threshold2, 255, cv2.THRESH_BINARY)
    
    # Define the region of interest (ROI)
    x, y, w, h = roi
    roi_diff = diff[y:y+h, x:x+w]
    
    # Apply a different threshold to the ROI
    _, roi_thresh = cv2.threshold(roi_diff, roithreshold, 255, cv2.THRESH_BINARY)
    
    # Insert the ROI threshold back into the thresholded image
    thresh1[y:y+h, x:x+w] = roi_thresh
    
    # Combine the thresholded images
    combined_thresh = cv2.bitwise_or(thresh1, thresh2)
    
    # Use morphological operations to highlight the differences
    kernel = np.ones((1, 1), np.uint8)
    diff_highlighted = cv2.dilate(combined_thresh, kernel, iterations=3)
    
    # Display the original images and the difference
    cv2.imshow("Image 1", image1)
    cv2.imshow("Aligned and Adjusted Image 2", adjusted_image2)
    cv2.imshow("Difference", diff)
    cv2.imshow("Highlighted Differences", diff_highlighted)
    
    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()
# Paths to the images to compare
image_path1 = '/run/media/shroomy/Media/BackUps/hackthefuture2024/files/good/images/pattern_image.png'
image_path2 = '/run/media/shroomy/Media/BackUps/hackthefuture2024/files/bad/rotation/c/images/0001.png'

# Define the region of interest (x, y, width, height)
roi = (0, 0, 0, 0)  # Example values, adjust as needed (x y width height)

# Define the thresholds
threshold1 = 60  # First threshold for the entire image
threshold2 = 60  # Second threshold for the entire image
roithreshold = 60 # ROI Threshold

# Compare the images
compare_images(image_path1, image_path2, threshold1, threshold2,roithreshold)


