from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_masks_and_save_large_regions(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Model setup
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_predictor = SamPredictor(sam)

    files_processed = False

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            files_processed = True
            print(f"Processing {filename}...")
            full_path = os.path.join(input_folder, filename)
            image = cv2.imread(full_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            mask_predictor.set_image(image_rgb)
            # Provide points as input prompt [X,Y]-coordinates
            input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
            input_label = np.array([1])

            # Predict the segmentation mask at that point
            masks, scores, logits = mask_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            significant_mask_found = False
            print(masks)

                    # Create a new blank image with a transparent background
            height, width = image_rgb.shape[:2]
            cropped_image = np.zeros((height, width, 4), dtype=np.uint8)  # Image with 4 channels (RGBA)
            for i in range(3):  # Copy RGB channels
                cropped_image[:, :, i] = image[:, :, i] * masks

                    # Set the alpha channel: 255 where there is a mask, 0 elsewhere
                cropped_image[:, :, 3] = masks * 255

                #final_image = cv2.cvtColor(cropped_image)

                # Save the image
                mask_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
                cv2.imwrite(mask_path, cropped_image)



    if not files_processed:
        logging.warning(f"No image files found in the input folder: {input_folder}")

# Example usage
input_folder = 'do_fused'
output_folder = 'segmented_images'
generate_masks_and_save_large_regions(input_folder, output_folder)

