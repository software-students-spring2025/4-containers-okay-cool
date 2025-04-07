"""
This script detects faces with MTCNN,
covering filling in their boundry boxes
and saves the redacted image.
"""

import time

import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN

# Starting timer
start_time = time.time()

# Read image input
IMAGE_NAME = "7"
IMAGE_PATH = f"./images/{IMAGE_NAME}.jpg"
IMAGE_PATH_PATCHED = f"./images/{IMAGE_NAME}-patched.jpg"
img = cv2.imread(IMAGE_PATH)  # pylint: disable=E1101

# MTCNN expects RGB, not BGR, so we are converting that here
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # pylint: disable=E1101

# Detect faces w/ MTCNN & extract their count
detector = MTCNN()
faces = detector.detect_faces(img_rgb)
NUM_FACES = 0
if faces:
    NUM_FACES = len(faces)

# Creating side-by-side sub-plots
fig, axs = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"wspace": 0.05})

# Put original image on plot
axs[0].imshow(img_rgb)
axs[0].axis("off")
axs[0].set_title("Original Image")

# Copy for processing
img_with_patches = img.copy()

for face in faces:
    x, y, w, h = face["box"]
    # Draw red borders
    img_with_patches = cv2.rectangle(  # pylint: disable=E1101
        img_with_patches,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),  # bgr format
        2,  # boundary box border width
    )
    # Draw blue fills
    img_with_patches = cv2.rectangle(  # pylint: disable=E1101
        img_with_patches, (x, y), (x + w, y + h), (0, 255, 255), -1  # -1 for fill
    )

# Convert patched image to RGB for matplotlib
img_with_patches_rgb = cv2.cvtColor(	# pylint: disable=E1101
    img_with_patches, cv2.COLOR_BGR2RGB	# pylint: disable=E1101
)

# Put patched image on plot
axs[1].imshow(img_with_patches_rgb)
axs[1].axis("off")
axs[1].set_title("Processed Image with MTCNN Faces")

# Save processed image to ./images
cv2.imwrite(IMAGE_PATH_PATCHED, img_with_patches)  # pylint: disable=E1101

end_time = time.time()
elapsed_time = end_time - start_time
rounded_time = round(elapsed_time, 1)

# Printing data
print(f"{NUM_FACES} FACES")
print(f"{rounded_time} SEC.")

# Display side-by-side plot
plt.show()
