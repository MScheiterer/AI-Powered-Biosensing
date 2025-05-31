import matplotlib.pyplot as plt
from segmentation.utils import show_masks
from PIL import Image
import numpy as np
from scipy.stats import linregress
import pandas as pd

def create_mask_plots(calibration_data, approx=False):
    """
    Visualizes masks for all images in a batch of calibration_data. 
    """
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


def create_intensity_plots(df_calibration: pd.DataFrame, well_sizes: list):
    n = len(well_sizes)
    
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]  # Ensure iterable

    for i, well_size in enumerate(well_sizes):
        draw_intensity_plot(axes[i], df_calibration, well_size)

    # Hide unused subplots, if any
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def draw_intensity_plot(axes: plt.Axes, df_calibration: pd.DataFrame, well_size: str):
    # Plot the calibration curve
    axes.errorbar(df_calibration['RI'], df_calibration[f'mean_intensity_{well_size}'],
                 yerr=df_calibration[f'std_intensity_{well_size}'], fmt='o', capsize=5,
                 color='blue', markersize=8, label='Measured Data')

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        df_calibration['RI'], df_calibration[f'mean_intensity_{well_size}']
    )
    r_squared = r_value**2
    x_fit = np.linspace(df_calibration['RI'].min(), df_calibration['RI'].max(), 100)
    y_fit = slope * x_fit + intercept
    axes.plot(x_fit, y_fit, '--', color='red',
             label=f'Linear Fit\ny = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.4f}')

    # Annotate points with NaCl percentages
    for i, row in df_calibration.iterrows():
        axes.annotate(f"{row['NaCl_percentage']}%",
                     (row['RI'], row[f'mean_intensity_{well_size}']),
                     xytext=(5, 5), textcoords='offset points')

    axes.set_title(f'Calibration Curve - {well_size} Wells')
    axes.set_xlabel('Refractive Index (RI)')
    axes.set_ylabel('Mean Intensity (AU/pixel)')
    axes.grid(False)
    axes.legend(loc="upper left")
    
