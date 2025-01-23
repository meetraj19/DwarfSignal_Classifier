import os
import shutil

# Paths to the original directories
image_directory = '/home/thales1/Dwarfsignal/images'
label_directory = '/home/thales1/Dwarfsignal/labels'

# reorganized images
output_directory = '/home/thales1/Dwarfsignal/organized'

# image and label filenames
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(label_directory) if f.endswith('.txt')]

#output directory
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each label file
for label_file in label_files:

    base_name = os.path.splitext(label_file)[0]

    # Construct full paths for the label file and corresponding image file
    label_path = os.path.join(label_directory, label_file)
    image_path = os.path.join(image_directory, base_name + '.jpg')

    # Check if the corresponding image file exists
    if os.path.exists(image_path):
        # Read the label
        with open(label_path, 'r') as file:
            label = file.read().strip()

        # Path to the subdirectory for the label
        label_dir = os.path.join(output_directory, label)

        # Ensure the subdirectory exists
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Destination path for the image
        destination_path = os.path.join(label_dir, base_name + '.jpg')

        # Copy the image to the new location
        shutil.copy(image_path, destination_path)
        print(f"Copied {image_path} to {destination_path}")
    else:
        print(f"No matching image found for {label_file}")



