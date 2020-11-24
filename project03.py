import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# ------------------------------------------------------------
# Step 1: Read Image
# -----------------------------------------------------------

# Read image
# Store it in the variable image
###
image = cv2.imread("CoinsA.png")
image2 = image.copy()
im = image.copy()
im2 = image.copy()
###

plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.show()

# ------------------------------------------------------------
# Step 2.1: Convert Image to Grayscale
# -----------------------------------------------------------

matplotlib.rcParams['image.cmap'] = 'gray'
imageCopy = image.copy()

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.subplot(122)
plt.imshow(imageGray);
plt.title("Grayscale Image");
plt.show()

# ------------------------------------------------------------
# Step 2.2: Split Image into R,G,B Channels
# -----------------------------------------------------------`

imageB = image[:, :, 0]
imageG = image[:, :, 1]
imageR = image[:, :, 2]

plt.figure(figsize=(20, 12))
plt.subplot(141)
plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.subplot(142)
plt.imshow(imageB);
plt.title("Blue Channel")
plt.subplot(143)
plt.imshow(imageG);
plt.title("Green Channel")
plt.subplot(144)
plt.imshow(imageR);
plt.title("Red Channel");
plt.show()

# ------------------------------------------------------------
# Step 3.1: Perform Thresholding
# -----------------------------------------------------------

ret, th1 = cv2.threshold(imageGray, 101, 255, cv2.THRESH_BINARY)
ret3, th3 = cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
ret, th2 = cv2.threshold(imageGray, 40, 255, cv2.THRESH_BINARY_INV)
print(ret3)

plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.subplot(132)
plt.imshow(th1);
plt.title("performing picture");
plt.subplot(133)
plt.imshow(th2);
plt.title("performing picture 2");
plt.show()

# ------------------------------------------------------------
# Step 3.2: Perform morphological operations
# -----------------------------------------------------------

image = cv2.imread("CoinsA.png", cv2.IMREAD_COLOR)

kernelSize = 9
kernelSize2 = 5

# Create Kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernelSize + 1, 2 * kernelSize + 1),
                                    (kernelSize, kernelSize))

element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernelSize2 + 1, 2 * kernelSize2 + 1),
                                     (kernelSize2, kernelSize2))
# Apply dilate function on the input image
# Perform Dilation
imDilated = cv2.dilate(th2, element)
imDilated2 = cv2.dilate(th2, element2)

plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.subplot(132)
plt.imshow(imDilated);
plt.title("Dilation");
plt.subplot(133)
plt.imshow(imDilated2);
plt.title("Dilation2");
plt.show()

kernelSize3 = 10
kernelSize4 = 4

element3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernelSize3 + 1, 2 * kernelSize3 + 1),
                                     (kernelSize3, kernelSize3))

element4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernelSize4 + 1, 2 * kernelSize4 + 1),
                                     (kernelSize4, kernelSize4))

imClose = cv2.erode(imDilated, element3)
imClose2 = cv2.erode(imDilated2, element4)

plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.subplot(132)
plt.imshow(imClose);
plt.title("Erosion");
plt.subplot(133)
plt.imshow(imClose2);
plt.title("Erosion2");
plt.show()




# ------------------------------------------------------------
# Step 3.3 Closing
# -----------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
thresh = cv2.adaptiveThreshold(th2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 1)
kernel = np.ones((4, 4), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                           kernel, iterations=4)
cont_img = closing.copy()
contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found = {}".format(len(contours)))

plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(image[:, :, ::-1]);
plt.title("Original Image")
plt.subplot(132)
plt.imshow(gray_blur);
plt.title("Erosion");
plt.subplot(133)
plt.imshow(closing);
plt.title("Erosion2");
plt.show()

# ------------------------------------------------------------
#Step 4.1: Create SimpleBlobDetector
# -----------------------------------------------------------
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# ------------------------------------------------------------
# Step 4.2: Detect Coins
# -----------------------------------------------------------
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 200
params.maxThreshold = 255

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(imClose)

contours, hierarchy = cv2.findContours(imClose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))

# ------------------------------------------------------------
# Step 4.3: Display the detected coins on original image
# -----------------------------------------------------------
cv2.drawContours(image, contours, -1, (0, 255, 0), 3);
for cnt in contours:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"] / M["m00"]))
    y = int(round(M["m01"] / M["m00"]))

    # Mark the center
    cv2.circle(image, (x, y), 10, (255, 0, 0), -1);

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(image[:, :, ::-1]);
plt.title(" Inside Contours an out side")
plt.show()

# ------------------------------------------------------------
# BLOBS
# -----------------------------------------------------------
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(imClose)
print("Number of contours found for Just inside = {}".format(len(keypoints)))

# ------------------------------------------------------------
# Step 4.3: Display the detected coins on original image
# -----------------------------------------------------------

for k in keypoints:
    x, y = k.pt
    x = int(round(x))
    y = int(round(y))
    # Mark center in BLACK
    cv2.circle(image2, (x, y), 5, (0, 0, 0), -1)
    # Get radius of blob
    diameter = k.size
    radius = int(round(diameter / 2))
    # Mark blob in RED
    cv2.circle(image2, (x, y), radius, (0, 255, 0), 2)

plt.figure(figsize=(12, 12))
plt.subplot(111)
plt.imshow(image2[:, :, ::-1]);
plt.title("Just Inside Circles")
plt.show()


# ------------------------------------------------------------
# Step 4.4: Perform Connected Component Analysis
# -----------------------------------------------------------


# Find connected components
_, imLabels = cv2.connectedComponents(th1)
plt.imshow(imLabels, cmap='gray')
plt.show()

# Display the labels
nComponents = imLabels.max()


# The following line finds the min and max pixel values
# and their locations on an image.
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)

# Normalize the image so that the min value is 0 and max value is 255.
imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)

# Convert image to 8-bits unsigned type
imLabels = np.uint8(imLabels)

# Apply a color map
imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(im[:, :, ::-1]);
plt.title("Original")
plt.subplot(122)
plt.imshow(imColorMap, cmap='gray');
plt.title("Connect Components")
plt.show()



# ------------------------------------------------------------
# Step 4.5: Detect coins using Contour Detection
# -----------------------------------------------------------

contours2, hierarchy2 = cv2.findContours(imClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found JUST EXTERNAL = {}".format(len(contours2)))

# Draw all contours
###
cv2.drawContours(im, contours2, -1, (0, 255, 0), 20);
for cnt in contours2:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"] / M["m00"]))
    y = int(round(M["m01"] / M["m00"]))

    # Mark the center
    cv2.circle(im, (x, y), 10, (255, 0, 0), -1);

plt.figure(figsize=(12, 12))
plt.subplot(111)
plt.imshow(im[:, :, ::-1]);
plt.title("JUST EXTERNAL")
plt.show()


for index, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
   #areaArray.pop(area)
    print("Contour #{} has area = {} and perimeter = {}".format(index + 1, area, perimeter))
    max_value = np.max(area)

print(max_value)

print("Maximum area = {}".format(area))


# ------------------------------------------------------------
# Step 4.5: Detect coins using Contour Detection
# "Take the last contour out"
# -----------------------------------------------------------

# Draw all contours
###
contours3 = []
contours3, hierarchy3 = cv2.findContours(imClose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found = {}".format(len(contours3)))
contours3.pop()
print("Number of contours found after pop it = {}".format(len(contours3)))

#cv2.drawContours(im2, contours3, -1, (0, 255, 0), 5);

for cnt in contours3:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"] / M["m00"]))
    y = int(round(M["m01"] / M["m00"]))

    # Mark the center
    cv2.circle(im2, (x, y), 10, (255, 0, 0), -1);

for k in contours3:
    #   x, y = k.pt
    M = cv2.moments(k)
    x = int(round(M["m10"] / M["m00"]))
    y = int(round(M["m01"] / M["m00"]))

    # Get radius of blob
    diameter = k.size
    radius = int(round(diameter / 6.2))
    # Mark blob in RED
    cv2.circle(im2, (x, y), radius, (0, 255, 0), 2)

plt.figure(figsize=(12, 12))
plt.subplot(111)
plt.imshow(im2[:, :, ::-1]);
plt.title("Just Inside Circles without last one")
plt.show()



