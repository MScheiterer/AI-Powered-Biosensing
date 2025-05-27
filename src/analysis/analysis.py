from scipy.ndimage import distance_transform_edt, map_coordinates
import numpy as np
import cv2
from PIL import Image
import os
from segmentation.sam2wrapper import SAM2Wrapper

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
        'masks_small': masks_small
    }


def compute_center(mask: np.ndarray) -> tuple[int, int]:
    y_coords, x_coords = np.nonzero(mask)
    return (int(round(y_coords.mean())), int(round(x_coords.mean())))


def distance_to_edge(mask: np.ndarray, center: tuple[int, int]) -> float:
    # Ensure binary mask
    binary_mask = mask.astype(bool)

    # Compute distance transform
    dist_map = distance_transform_edt(binary_mask)

    # Extract coordinates (ensure float and inside bounds)
    y, x = center
    
    # plt.imshow(mask)
    # plt.scatter([x], [y], color='cyan')
    # plt.show()

    # Interpolate distance at floating-point center
    return float(map_coordinates(dist_map, [[y], [x]], order=1, mode='nearest')[0])


def calculate_intensity(gray_image, masks):
    circle_masks = []
    for mask in masks:
        m = mask["segmentation"].astype(np.uint8)

        center = compute_center(m)
        radius = int(round(0.75 * distance_to_edge(m, center)))
        print(f"center: {center}, radius: {radius}")

        circle_mask = np.zeros(m.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, (center[1], center[0]), radius, 255, -1) # numpy uses (y, x), cv2 uses (x, y)
        circle_masks.append(circle_mask)

    # Assuming gray_image is an RGB or RGBA numpy array
    # gray_image = np.stack([gray_image]*3, axis=-1).astype(np.uint8)  # convert grayscale to RGB
    # gray_image = np.concatenate([gray_image, 255 * np.ones_like(gray_image[..., :1])], axis=-1)  # Add alpha

    # for mask in circle_masks:
    #     gray_image[mask] = (255, 0, 255, 0.5)
    #     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #     # Try to smooth contours
    #     contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    #     cv2.drawContours(gray_image, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    # plt.figure(figsize=(8, 6))
    # plt.imshow(gray_image)
    # plt.axis('off')
    # plt.title("Image with Mask Overlay")
    # plt.show()
        
    intensities = [np.mean(gray_image[circle_mask]) for circle_mask in circle_masks]
    return intensities, circle_masks