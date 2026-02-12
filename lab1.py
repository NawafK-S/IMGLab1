
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR,"images")
OUT_DIR = os.path.join(BASE_DIR,"output")
os.makedirs(OUT_DIR,exist_ok=True)

img = cv2.imread(os.path.join(IMG_DIR,"cameraman.tif"),cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap=cm.Greys_r)
plt.title("OpenCV Image")
plt.axis("off")
plt.show()

img2 = Image.open(os.path.join(IMG_DIR,"lena_gray_256.tif"))

plt.imshow(img2,cmap=cm.Greys_r)
plt.title("PIL Image")
plt.axis("off")
plt.show()

cv2.imwrite(os.path.join(OUT_DIR,"new_image.jpg"),img)
img2.save(os.path.join(OUT_DIR,"new_image2.jpg"))

print("Shape:",img.shape)
print(img[:10,:10])

img_array = np.array(img2)
print(img_array.shape)
print(img_array[:10,:10])
