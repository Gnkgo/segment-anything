from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import cv2
import os

def generate_masks_and_save_large_regions(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Model setup
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        min_mask_region_area=2,  # Minimum area of regions to consider as mask
    )

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")
            full_path = os.path.join(input_folder, filename)
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Generate masks
            masks = mask_generator.generate(image)
            image_area = image.shape[0] * image.shape[1]

            largest_mask = None
            largest_mask_area = 0

            # Find the largest mask
            for mask in masks:
                mask_array = mask['segmentation'].astype(np.uint8)
                mask_area = np.sum(mask_array)
                
                if mask_area > largest_mask_area:
                    largest_mask_area = mask_area
                    largest_mask = mask_array

            # Process the largest mask if it exists
            if largest_mask is not None:
                height, width = largest_mask.shape
                cropped_image = np.zeros((height, width, 4), dtype=np.uint8)  # Image with 4 channels (RGBA)
                
                for i in range(3):  # Copy RGB channels
                    cropped_image[:, :, i] = image[:, :, i] * largest_mask

                # Set the alpha channel: 255 where there is a mask, 0 elsewhere
                cropped_image[:, :, 3] = largest_mask * 255

                final_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA)

                # Save the image
                mask_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_largest_mask.png")
                cv2.imwrite(mask_path, final_image)

# Example usage
input_folder = 'background_and_nipple'
output_folder = 'nipple'
generate_masks_and_save_large_regions(input_folder, output_folder)
