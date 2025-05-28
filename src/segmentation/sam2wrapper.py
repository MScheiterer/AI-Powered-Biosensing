"""
Wrapper class to streamline the use of SAM2AutomaticMaskGenerator:
    - Parametrization and creation
    - Automatic mask generation
    - Filtering 
    - Displaying intermediate and final masks
"""

import torch
from . import utils
import os
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Wrapper:
    """
    Attributes:
        self.config:
        self.checkpoint:
        self.model:
        self.image:
        self.masks:
        self.mask_data:
    """
    def __init__(self, config=None, checkpoint=None, points_per_side=128):
        """
        Initializes the SAM2 model and predictor.
        
        Parameters:
        -----------
        config : .yaml SAM2 configuration file. (Default sam2.1_s.yaml)
        checkpoint : .pt SAM2 checkpoint file. (Default sam2.1_small.pt)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"==== Using device: {device} ====")

        base_path = os.getcwd() # Save cwd path
        os.chdir(f"{os.path.dirname(os.path.abspath(__file__))}/../../sam2")
    
        if config is None:
            config = "/configs/sam2.1/sam2.1_hiera_l.yaml" # @param ["sam2.1_hiera_t.yaml", "sam2.1_hiera_s.yaml", "sam2.1_hiera_b+.yaml", "sam2.1_hiera_l.yaml"]
        if checkpoint is None:
            checkpoint = "C:\\Users\\Micha\\Desktop\\BachelorProject\\AI-Powered-Biosensing\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
            #checkpoint = "checkpoints/sam2.1_hiera_small.pt"  # @param ["sam2.1_hiera_tiny.pt", "sam2.1_hiera_small.pt", "sam2.1_hiera_base_plus.pt", "sam2.1_hiera_large.pt"]
            
        os.chdir(base_path) # Restore cwd path
        
        self.config = config
        self.checkpoint = checkpoint
        self.model = build_sam2(config, checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side
        )
        print("==== SAM2AutomaticMaskGenerator initialized ====")

    def plot_image(self):
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('off')
            plt.show()
        else:
            print("Image must be set first.")

    def generate_masks(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = img
        self.masks = self.mask_generator.generate(img)

    def filter_masks(self):
        height, width = self.image.shape[:2]
        total_pixels = height * width
        self.masks = [mask for mask in self.masks if 0.00015 < mask["area"] / total_pixels < 0.005]
        self.masks = [mask for mask in self.masks if utils.compute_circularity(mask["segmentation"]) > 0.75]

        masks_large = [mask for mask in self.masks if mask['area'] / total_pixels > 0.003]
        masks_medium = [mask for mask in self.masks if 0.0005 < mask['area'] / total_pixels < 0.001]
        masks_small = [mask for mask in self.masks if mask['area'] / total_pixels <= 0.0005]

        return self.masks, masks_large, masks_medium, masks_small

    def visualize_masks(self, masks=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        utils.show_anns(self.masks)
        if masks:
            utils.show_anns(masks, color=(255, 0, 0, 0.5))
        plt.axis('off')
        plt.show()