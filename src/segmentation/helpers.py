"""
Contains helper functions for the SAM2Predictor and SAM2AutomaticMaskGenerator Wrapper classes
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

np.random.seed(3)


# Mask visualization for AMG masks
def show_anns(anns, ax=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None: 
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([(0, 255, 0), [0.5]])
        img[m] = color_mask 
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def show_mask(mask, ax, borders = True):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)      

def show_masks(image, masks, scores, max_count, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i > max_count:
            return
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 

# Function to filter masks based on circularity and area
def filter_masks(masks, image, area_filter, min_circularity=0.8):
    filtered_masks = {
        'big': [],
        'medium': [],
        'small': []
    }
    circular_masks = []
    
    # Filter by circularity
    if masks:
        for mask in masks:
            circularity = compute_circularity(mask['segmentation'])
            if (circularity >= min_circularity):
                circular_masks.append(mask)
        print(f"Circular masks: {len(circular_masks)}")
        if circular_masks: 
            filtered_masks = filter_by_area(circular_masks, area_filter[0], area_filter[1], area_filter[2])
            
    return filtered_masks


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

    
def filter_by_area(masks, big_wells, medium_wells, small_wells):
    big_masks, medium_masks, small_masks = [], [], []
    filtered_masks = {
        'big': big_masks,
        'medium': medium_masks,
        'small': small_masks
    }
    
    # Filter by area - assume no more than 15% deviation within well group 
    if big_wells:
        lmax_area = max(mask['area'] for mask in masks)
        lmin_area = 0.80 * lmax_area
        for mask in masks:
            if (mask['area'] >= lmin_area):
                big_masks.append(mask)

    # Assume medium wells ca. 1/6 of large wells
    if medium_wells:
        mmax_area = 0.16 * lmax_area
        mmin_area = 0.16 * lmin_area
        for mask in masks:
            if (mask['area'] >= mmin_area and mask['area'] <= mmax_area):
                medium_masks.append(mask)
                
    if small_wells:
        smin_area = min(mask['area'] for mask in masks)
        smax_area = 1.2 * smin_area
        # Check whether medium = small well (if no small well was detected)
        if smin_area < 0.5 * mmin_area:
            for mask in masks:
                if (mask['area'] >= smin_area and mask['area'] <= smax_area):
                    small_masks.append(mask)
    
    print(f"Filtered masks: {(len(big_masks) + len(medium_masks) + len(small_masks))}")
    return filtered_masks