import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mtcnn import MTCNN
import time

# starting timer
start_time = time.time()

# read image input
name = "image"
path = f"./images/{name}.jpg"
path_patched = f"./images/{name}-patched.jpg"
img = cv2.imread(path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MTCNN expects RGB

# detect faces w/ MTCNN
detector = MTCNN()
faces = detector.detect_faces(img_rgb)
num_faces = 0
if faces:
	num_faces = len(faces)

# side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'wspace': 0.05})

# show original image on plot
axs[0].imshow(img_rgb)
axs[0].axis('off')
axs[0].set_title("Original Image")

# copy for processing
img_with_patches = img.copy()

for face in faces:
	x, y, w, h = face['box']
	# draw red borders
	img_with_patches = cv2.rectangle(
	    img_with_patches, 
	    (x, y), 
	    (x + w, y + h), 
	    (0, 0, 255),	# bgr format
	    2   # boundry box border width
	)
	# draw blue fills
	img_with_patches = cv2.rectangle(
	    img_with_patches, 
	    (x, y), 
	    (x + w, y + h), 
	    (0, 255, 255),  # Blue color in BGR format
	    -1  # -1 for fill
	)

# convert patched image to RGB for matplotlib
img_with_patches_rgb = cv2.cvtColor(img_with_patches, cv2.COLOR_BGR2RGB)

# show patched image on plot
axs[1].imshow(img_with_patches_rgb)
axs[1].axis('off')
axs[1].set_title("Processed Image with MTCNN Faces")

# save processed image to ./images directory
cv2.imwrite(path_patched, img_with_patches)

end_time = time.time()
elapsed_time = end_time - start_time
rounded_time = round(elapsed_time, 1)

# printing data
print(f"{num_faces} FACES")
print(f"{rounded_time} SEC.")

# display side by side plot
plt.show()