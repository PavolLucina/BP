import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

def transform_and_save_images(original_dir, transformed_dir):
    """
    Transforms images from the original directory and saves them to the transformed directory
    while maintaining the directory structure and filenames.

    Parameters:
    - original_dir (str or Path): Path to the original MacaqueFaces directory.
    - transformed_dir (str or Path): Path to the transformed MacaqueFacesTransformed directory.
    """
    original_path = Path(original_dir)
    transformed_path = Path(transformed_dir)

    # Define the transformation: Resize to [224, 224]
    transform = T.Resize([224, 224])

    # Check if the original directory exists
    if not original_path.exists() or not original_path.is_dir():
        print(f"Error: The original directory '{original_dir}' does not exist or is not a directory.")
        return

    # Create the transformed directory if it doesn't exist
    if not transformed_path.exists():
        transformed_path.mkdir(parents=True)
        print(f"Created transformed directory: {transformed_path}")

    # Traverse the original directory recursively
    for root, dirs, files in os.walk(original_path):
        # Compute the relative path from the original root
        rel_path = Path(root).relative_to(original_path)
        # Define the corresponding path in the transformed directory
        transformed_subdir = transformed_path / rel_path

        # Create the subdirectory in the transformed directory if it doesn't exist
        if not transformed_subdir.exists():
            transformed_subdir.mkdir(parents=True)
            print(f"Created subdirectory: {transformed_subdir}")

        for file in files:
            original_file_path = Path(root) / file
            transformed_file_path = transformed_subdir / file

            try:
                # Open the original image
                with Image.open(original_file_path) as img:
                    # Apply the transformation
                    transformed_img = transform(img)

                    # Preserve the original image format
                    img_format = img.format  # e.g., 'JPEG', 'PNG'

                    # Save the transformed image with the same filename and format
                    transformed_img.save(transformed_file_path, format=img_format)

                print(f"Transformed and saved: {transformed_file_path}")

            except Exception as e:
                print(f"Error processing '{original_file_path}': {e}")

    print("All images have been transformed and saved successfully.")

if __name__ == "__main__":
    # Define the paths to the original and transformed directories
    original_directory = 'data/MacaqueFaces/MacaqueFaces'  # Update this path if different
    transformed_directory = 'data/MacaqueFacesTransformed/MacaqueFaces'  # Update this path if different

    transform_and_save_images(original_directory, transformed_directory)
