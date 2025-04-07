"""
Script to create a sample test image for the face redaction client.
"""

import os
import cv2
import numpy as np

# Create a blank image
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (255, 255, 255)  # White background

# Draw a simple face-like shape
# Head
cv2.circle(img, (300, 200), 100, (200, 200, 200), -1)
# Eyes
cv2.circle(img, (250, 170), 15, (0, 0, 0), -1)
cv2.circle(img, (350, 170), 15, (0, 0, 0), -1)
# Mouth
cv2.ellipse(img, (300, 220), (40, 20), 0, 0, 180, (0, 0, 0), 3)

# Save the image to the input directory
os.makedirs("images/input", exist_ok=True)
cv2.imwrite("images/input/sample.jpg", img)

print("Sample image created at images/input/sample.jpg") 