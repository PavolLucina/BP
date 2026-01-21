from PIL import Image
import os

# Define input and output directories
input_dir = "data\\atrw_reid_train\\train"
output_dir = "data\\atrw_reid_train\\train_ciernobiele"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to black and white (grayscale)
            bw_img = img.convert("L")
            # Save to output directory
            bw_img.save(output_path)
            print(f"Converted: {filename} -> {output_path}")
    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("Conversion complete!")
