import matplotlib.pyplot as plt
from segmentation.utils import show_masks
from PIL import Image

def create_mask_plots(calibration_data, approx=False):
    n = len(calibration_data)
    
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]  # Ensure iterable

    for i, elem in enumerate(sorted(calibration_data, key=lambda x: x["NaCl_percentage"])):
        if approx:
            masks_large = elem["approx_masks_large"]
            masks_medium = elem["approx_masks_medium"]
            masks_small = elem["approx_masks_small"]
        else:
            masks_large = elem["masks_large"]
            masks_medium = elem["masks_medium"]
            masks_small = elem["masks_small"]
        img_name = elem["image_name"]
        img = Image.open(elem["image_path"])

        axes[i].imshow(img)
        show_masks(axes[i], masks_large, color=(255, 0, 0, 0.5))
        show_masks(axes[i], masks_medium, color=(0, 255, 0, 0.5))
        show_masks(axes[i], masks_small, color=(0, 0, 255, 0.5))
        axes[i].set_title(f"{img_name} - {elem['NaCl_percentage']}%")
        
    # Hide unused subplots, if any
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


