import os
from PIL import Image

folder_path = "dataset/"
file_paths = []
images = []
new_size = (64, 64)

# Iterate over all files in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        Image.open(os.path.join(root, file)).resize(new_size).save("resizedData/" + file)
        print("Saved Image: " + file)
