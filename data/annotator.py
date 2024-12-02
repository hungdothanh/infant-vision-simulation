
import os
import pandas as pd

# Define the folder containing the images
image_folder = './dataset/images'

# Define where to save the annotation file
output_folder = './dataset/labels'
os.makedirs(output_folder, exist_ok=True)  # Create labels folder if it doesn't exist

# Initialize a list to store the annotations
annotations = []

# Loop through the files in the image folder
for file_name in os.listdir(image_folder):
    if file_name.lower().startswith("cat"):
        label = 0
    elif file_name.lower().startswith("dog"):
        label = 1
    else:
        continue  # Skip files that don't start with 'cat' or 'dog'
    
    # Exclude the file extension (.jpg) from the name
    name_without_extension = os.path.splitext(file_name)[0]
    annotations.append({"image_name": name_without_extension, "label": label})

# Convert the annotations to a DataFrame
annotations_df = pd.DataFrame(annotations)

# Save the DataFrame to a CSV file
output_csv = os.path.join(output_folder, 'labels.csv')
annotations_df.to_csv(output_csv, index=False)

print(f"Annotation file saved as {output_csv}")
