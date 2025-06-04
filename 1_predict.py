from ultralytics import YOLO
import cv2
import numpy as np
import os

# Define paths
model_path = r'C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\Model.pt'
folder_path = r'C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\Testset'
output_overlay_folder = r'C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\overlay'
output_mask_folder = r'C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\mask'
labelmap_path = r'C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\labelmap.txt'

# Ensure output directories exist
os.makedirs(output_overlay_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Load the YOLO model
model = YOLO(model_path)

# Load label map (label index -> color)
def load_labelmap(file_path):
    label_map = {}
    with open(file_path, "r") as file:
        for line in file.readlines():
            if ":" in line and not line.startswith("#"):
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    label_id = parts[0]

                    # Skip non-numeric labels (like "background")
                    if not label_id.isdigit():
                        continue

                    color = tuple(map(int, parts[1].split(",")))
                    label_map[int(label_id)] = color  # Convert label_id to int

    return label_map

# Load label-to-color mapping
label_colors = load_labelmap(labelmap_path)

# Confidence threshold (adjust if necessary)
CONFIDENCE_THRESHOLD = 0.4

# Process all images in the folder
for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)

    # Read the input image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping {img_name} (could not read)")
        continue  # Skip unreadable files

    H, W, _ = img.shape

    # Run YOLO model inference
    results = model(img)

    # Process results safely
    for result in results:
        if result.masks is None or result.boxes is None:
            print(f"⚠️ No objects detected in {img_name}!")
            continue  # Skip processing if no mask

        img_with_mask = img.copy()
        mask_only_image = np.zeros_like(img)

        # Find the detection with the highest confidence
        max_conf_index = np.argmax(result.boxes.conf.cpu().numpy())  # Get index of highest confidence
        max_conf_score = result.boxes.conf[max_conf_index].item()  # Highest confidence score

        if max_conf_score < CONFIDENCE_THRESHOLD:
            print(f"⚠️ Skipping {img_name}, no detection meets confidence threshold")
            continue  # Skip image if no confident detection

        # Extract the highest-confidence mask
        mask = result.masks.data[max_conf_index].numpy() * 255
        mask = cv2.resize(mask, (W, H))
        mask = mask.astype(np.uint8)

        # Get the highest-confidence class ID
        class_id = int(result.boxes.cls[max_conf_index].item())

        # Get the color from labelmap
        color = label_colors.get(class_id, (255, 255, 255))  # Default white if not found
        color_mask = np.zeros_like(img)
        color_mask[:, :] = color  # Apply class-specific color

        # Apply the mask only to object pixels
        object_pixels = mask > 0  # Get only the detected object
        img_with_mask[object_pixels] = cv2.addWeighted(img[object_pixels], 0.7, color_mask[object_pixels], 0.3, 0)

        # Create a separate mask image (black background + object only in color)
        mask_only_image[object_pixels] = color_mask[object_pixels]

        # Draw bounding box and label text
        x1, y1, x2, y2 = result.boxes.xyxy[max_conf_index].cpu().numpy().astype(int)
        label_text = f"Class {class_id} ({max_conf_score:.2f})"

        cv2.rectangle(img_with_mask, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_with_mask, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Create class-specific folder inside the overlay directory
        class_folder = os.path.join(output_overlay_folder, f"{class_id}")
        os.makedirs(class_folder, exist_ok=True)  # Ensure class folder exists

        # Define output filenames
        base_name = os.path.splitext(img_name)[0]
        output_overlay_path = os.path.join(class_folder, f"{base_name}.png")
        output_mask_path = os.path.join(output_mask_folder, f"{base_name}.png")

        # Save images
        cv2.imwrite(output_overlay_path, img_with_mask)
        cv2.imwrite(output_mask_path, mask_only_image)

        print(f"✅ Saved overlay to {output_overlay_path}")
        print(f"✅ Saved mask to {output_mask_path}")

print("🎉 Batch processing complete! All images processed.")

