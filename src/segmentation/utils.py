"""
Contains helper functions for the SAM2AutomaticMaskGenerator Wrapper class
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


# Mask visualization for AMG masks
def show_masks(masks, ax=None, color=(0, 255, 0, 0.5)):
    """
    anns need to have the keys "segmentation" and "area".
    """
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    if ax is None: 
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for mask in sorted_masks:
        m = mask['segmentation']
        img[m] = color
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


# Function to calculate circularity of a contour
def compute_circularity(mask):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0  # No valid contour found

    contour = max(contours, key=cv2.contourArea)  # Get the largest contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:  # Avoid division by zero
        return 0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity
    
    
def compare_masks(axes, i, masks, image):
    return NotImplementedError("This function is not implemented yet. Please implement the compare_masks function.")