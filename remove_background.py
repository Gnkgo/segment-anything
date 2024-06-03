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

            # Process each mask
            for idx, mask in enumerate(masks):
                mask_array = mask['segmentation'].astype(np.uint8)
                mask_area = np.sum(mask_array)

                # Check if the mask area is greater than half the image area
                if mask_area > image_area / 4:
                    # Create a new blank image with a transparent background
                    height, width = mask_array.shape
                    cropped_image = np.zeros((height, width, 4), dtype=np.uint8)  # Image with 4 channels (RGBA)
                    for i in range(3):  # Copy RGB channels
                        cropped_image[:,:,i] = image[:,:,i] * mask_array

                    # Set the alpha channel: 255 where there is a mask, 0 elsewhere
                    cropped_image[:,:,3] = mask_array * 255
                    
                    final_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA)

                    # Save the image
                    mask_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask_{idx}.png")
                    cv2.imwrite(mask_path, final_image)

                    # Save the image


# Example usage
input_folder = 'removed_background_fused'
output_folder = 'segmented_images'
generate_masks_and_save_large_regions(input_folder, output_folder)
