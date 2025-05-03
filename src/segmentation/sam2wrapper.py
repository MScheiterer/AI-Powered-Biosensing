"""
Wrapper class to simplify SAM2AutomaticMaskGenerator and SAM2Predictor functionality:
    - Parametrization and creation
    - Automatic mask generation or segmentation with a single or multiple prompts
    - Filtering and Stitching together multiple masks 
    - Displaying intermediate and final masks and prompts
"""

import torch
import helpers
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from abc import abstractmethod
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
    @abstractmethod
    def __init__(self, config, checkpoint):
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
        print(f"using device: {device}")

        base_path = os.getcwd() # Save cwd path
        os.chdir(f"{os.path.dirname(os.path.abspath(__file__))}/segment-anything-2")
    
        if config is None:
            config = f"/configs/sam2/sam2_hiera_s.yaml" # @param ["sam2.1_hiera_t.yaml", "sam2.1_hiera_s.yaml", "sam2.1_hiera_b+.yaml", "sam2.1_hiera_l.yaml"]
        if checkpoint is None:
            checkpoint = f"checkpoints/sam2_hiera_small.pt"  # @param ["sam2.1_hiera_tiny.pt", "sam2.1_hiera_small.pt", "sam2.1_hiera_base_plus.pt", "sam2.1_hiera_large.pt"]
            
        os.chdir(base_path) # Restore cwd path
        
        self.config = config
        self.checkpoint = checkpoint
        self.model = build_sam2(config, checkpoint, device=device)

    @abstractmethod
    def visualize_masks():
        pass

    def plot_image(self):
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('off')
            plt.show()
        else:
            print("Image must be set first.")

    def set_image(self, image, all_positive=None, prompts=None):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = img

    @abstractmethod
    def generate_masks(self):
        pass


class AutomaticMaskGenerator(SAM2Wrapper):
    """
    Wrapper class to simplify usage of the SAM2 Automatic Mask Generator. Handles inititalizing SAM2, mask segmentation and visualization. 
    
    Attributes:
        self.mask_generator:

    Methods:

    """
    def __init__(self, config=None, checkpoint=None):
        super().__init__(config, checkpoint)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True
        )
        print("-- SAM2amg says: Hello World!")

    def generate_masks(self):
        self.mask_data = self.mask_generator.generate(self.image)
        self.masks = [mask_info["segmentation"] for mask_info in self.mask_data]


    def visualize_masks(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        #helpers.show_masks(self.image, self.masks, self.scores)
        helpers.show_anns(self.mask_data)
        plt.axis('off')
        plt.show()


class Predictor(SAM2Wrapper):
    """
    Wrapper class to simplify usage of the SAM2 Image Predictor. Handles inititalizing SAM2, mask segmentation, filtering,
    stitching and visualization. 

    Attributes:
        self.predictor:
        self.prompts:
        self.labels:
        self.masks:
        self.scores:
        self.mask_data:
    """
    def __init__(self, config=None, checkpoint=None):
        super().__init__(config, checkpoint)
        self.predictor = SAM2ImagePredictor(self.model)
        print("-- SAM2predictor says: Hello World!")

    def set_image(self, image, prompts, all_positive=True, labels=None):
        """
        Sets the image and corresponding prompts and labels for the SAM2 predictor. 
       
        Parameters:
        -----------
        image : numpy.ndarray
           Input image on which to perform blob detection.
        prompts : list
           Coordinates for segmentation in the form [[a,b], [c,d], ...]
        all_positive : bool, default=True
           If true, positive labels will be generated for each prompt. If false, labels must be provided
        labels : list
            List of labels corresponding to each prompt, 1 = positive, 0 = negative. Form: [[a], [b], ...]
        
        Notes:
        ------
        SAM2 handles multiple points provided to segment an individual object (e.g. a negative point to exclude the background. 
        This functionality is not yet implemented.
        """
        super().set_image(image)
        self.predictor.set_image(image)

        # Format prompts 
        self.prompts = np.array([[pt] for pt in prompts])
        
        # Format labels
        if all_positive:
            self.labels = np.array([[pt] for pt in np.ones(len(prompts), dtype=np.int32)])
        else:
            self.labels = np.array([[pt] for pt in labels])
    
    def predict(self):
        # Run inference
        print("predicting...")
        masks, scores, logits = self.predictor.predict(
            point_coords=self.prompts,
            point_labels=self.labels,
            multimask_output=True
        )
        self.masks = masks
        self.scores = scores
        #results = []
        #for i, multimask in enumerate(masks):
            #best_mask_idx = np.argmax(scores[i])
            #best_mask = multimask[best_mask_idx]
            #best_score = scores[i][best_mask_idx]
    
            #results.append({
                #"masks": masks,
                #"scores": scores,
                #"best_mask": best_mask,
                #"best_score": best_score
            #})
            #self.masks = [result['best_mask'] for result in results]
            #self.scores = [result['best_score'] for result in results]

            #helpers.show_masks(self.image, multimask, scores[i], 20)
                

    def generate_masks(self, min_area, max_area):
        """
        Combine the best masks from multiple prompts into a single visualization.
            
        Returns:
            combined_mask: A single image with all selected masks overlaid with different colors
            mask_info: Information about which masks were selected and their properties
        """
        self.predict()

        # To calculate area as a percentage
        height, width = self.image.shape[:2]
        total_pixels = height * width
        mask_data = []

        # multimask_output=True yields 3 masks per prompt, so a filter must choose best mask for the prompt
        for i, (multimask, multiscore) in enumerate(zip(self.masks, self.scores)):
            for (mask, score) in zip(multimask, multiscore):
                area = np.sum(mask) / total_pixels  # Area as a percentage
                if min_area <= area <= max_area:  # Filter by area
                    mask_data.append({
                        'prompt_idx': i,
                        'mask': mask,
                        'score': score,
                        'area': area
                    })
                
        self.masks = [mask['mask'] for mask in mask_data]
        self.scores = [mask['score'] for mask in mask_data]
        self.mask_data = mask_data

    def visualize_masks(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)   
        for mask in self.masks:
            helpers.show_mask(mask, plt.gca())
        helpers.show_points(self.prompts, self.labels, plt.gca())

    def visualize_masks_individually(self, max_count):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        helpers.show_masks(self.image, self.masks, self.scores, max_count)
        plt.axis('off')
        plt.show()