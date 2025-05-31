from scipy.ndimage import distance_transform_edt, map_coordinates
import numpy as np
import cv2
from PIL import Image
import os
from segmentation.sam2wrapper import SAM2Wrapper
from segmentation.utils import show_masks
from . import utils
import pandas as pd

### Remove Outliers? ###
# Remove outliers using a 2σ filter
#mean_val = np.mean(intensities)
#std_val = np.std(intensities)
#lower_thresh = mean_val - 2 * std_val
#upper_thresh = mean_val + 2 * std_val
#valid_indices = [i for i, intensity in enumerate(intensities)
                    #if lower_thresh <= intensity <= upper_thresh]
#intensities = [intensities[i] for i in valid_indices]
#masks = [masks[i] for i in valid_indices]

def run_analysis(data: list[dict], model: SAM2Wrapper) -> list[dict]:
    """
    Runs the analyis pipeline: intensity analysis (with option to include mask recalculation), 
        mask visualization, plot visualization, result export.

    Args:
        data (list[dict]): image data returned by data_preprocessing.
    """
    for group, items in data.items():
        calibration_data = []
        for item in items:
            calibration_data.append(intensity_analysis(item, model))
            
            
        ### Pandas DataFrame Creation and clean up ###
        df_data = pd.DataFrame(calibration_data).sort_values('NaCl_percentage')
        df_data = df_data.drop(columns=['masks', 'masks_large', 'masks_medium', 'masks_small'])
        # Fill NaN values with 0 if the entire column is NaN
        df_data.loc[:, df_data.isna().all()] = 0 
        # Compute refractive index (RI) using a simplified linear model
        df_data['RI'] = 1.3330 + 0.0018 * df_data['NaCl_percentage']
        # Drop rows with NaN values
        df_calibration = df_data.dropna()
        
        
        ### Mask Visualization ###
        utils.create_mask_plots(calibration_data)
        utils.create_mask_plots(calibration_data, approx=True)
        
        
        ### Intensity vs RI Plots ###
        utils.create_intensity_plots(df_calibration)
        
        ### Save DataFrame to CSV and plots as one pdf file ###



def intensity_analysis(item: dict, model: SAM2Wrapper) -> dict:
    image_path = item["file_path"]
    image = np.array(Image.open(image_path))
    model.generate_masks(image)
    masks, masks_large, masks_medium, masks_small = model.filter_masks()
    
    # Create grayscale version for intensity measurement
    gray_image = np.array(Image.open(image_path).convert("L"))

    # Compute mean intensity for each filtered mask
    intensities_all, approx_masks_all = calculate_intensity(gray_image, masks)
    intensities_small, approx_masks_small = calculate_intensity(gray_image, masks_small)
    intensities_medium, approx_masks_medium = calculate_intensity(gray_image, masks_medium)
    intensities_large, approx_masks_large = calculate_intensity(gray_image, masks_large)

    mean_intensity_large = np.mean(intensities_large)
    std_intensity_large = np.std(intensities_large)
    mean_intensity_medium = np.mean(intensities_medium)
    std_intensity_medium = np.std(intensities_medium)
    mean_intensity_small = np.mean(intensities_small)
    std_intensity_small = np.std(intensities_small)
    mean_intensity_all = np.mean(intensities_all)
    std_intensity_all = np.std(intensities_all)
    
    return {
        'NaCl_percentage': item["nacl_percentage"],
        'mean_intensity_large': mean_intensity_large,
        'std_intensity_large': std_intensity_large,
        'mean_intensity_medium': mean_intensity_medium,
        'std_intensity_medium': std_intensity_medium,
        'mean_intensity_small': mean_intensity_small,
        'std_intensity_small': std_intensity_small,
        'mean_intensity_all': mean_intensity_all,
        'std_intensity_all': std_intensity_all,
        'num_wells': len(masks),
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'masks': masks,
        'masks_large': masks_large,
        'masks_medium': masks_medium,
        'masks_small': masks_small,
        'approx_masks': approx_masks_all,
        'approx_masks_large': approx_masks_large,
        'approx_masks_medium': approx_masks_medium,
        'approx_masks_small': approx_masks_small,
    }


def compute_center(mask: np.ndarray) -> tuple[int, int]:
    y_coords, x_coords = np.nonzero(mask)
    return (int(round(y_coords.mean())), int(round(x_coords.mean())))


def distance_to_edge(mask: np.ndarray, center: tuple[int, int]) -> float:
    binary_mask = mask.astype(bool)
    dist_map = distance_transform_edt(binary_mask)
    y, x = center
    return float(map_coordinates(dist_map, [[y], [x]], order=1, mode='nearest')[0])


def calculate_intensity(gray_image, masks):
    circle_masks = []
    for mask in masks:
        m = mask["segmentation"].astype(np.uint8)

        center = compute_center(m)
        radius = int(round(0.75 * distance_to_edge(m, center)))

        circle_mask = np.zeros(m.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, (center[1], center[0]), radius, 255, -1) # numpy uses (y, x), cv2 uses (x, y)
        circle_masks.append(circle_mask)
        
    intensities = [np.mean(gray_image[circle_mask]) for circle_mask in circle_masks]
    return intensities, circle_masks