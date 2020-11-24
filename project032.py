import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

image = cv2.imread("CoinsB.png")
im = cv2.imread("CoinsB.png")
matplotlib.rcParams['image.cmap'] = 'gray'
imageCopy = image.copy()
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.subplot(122)
plt.imshow(imageGray);
plt.title("Grayscale Image");
plt.show()


imageB = image[:, :, 0]
imageG = image[:, :, 1]
imageR = image[:, :, 2]
cv2.imwrite('imageB.png',imageB )

plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(image[:,:,::-1]);
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

# --------------------------------
# Step 3.1: Perform Thresholding
#-----------------------------------

ret, th1 = cv2.threshold(imageB, 133, 255, cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(imageB, 115, 255, cv2.THRESH_BINARY_INV)



plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(th1);
plt.title("performing picture");
plt.subplot(122)
plt.imshow(th2);
plt.title("performing picture 2");
plt.show()

# ------------------------------------------------------------
# Step 3.2: Perform morphological operations OPEN
# -----------------------------------------------------------


openingSize = 3

# Selecting a elliptical kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * openingSize + 1, 2 * openingSize + 1),
            (openingSize,openingSize))

imageMorphOpened = cv2.morphologyEx(th1, cv2.MORPH_OPEN,
                        element,iterations=3)

# ----------------------------------------------------------
openingSize2 = 10

# Selecting a elliptical kernel
element2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * openingSize2 + 1, 2 * openingSize2 + 1),
            (openingSize2,openingSize2))

imageMorphOpened2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN,
                        element2,iterations=3)



plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(imageMorphOpened);
plt.title("MorphoOpen");
plt.subplot(122)
plt.imshow(imageMorphOpened2);
plt.title("MorphoOpen2");
plt.show()

# ------------------------------------------------------------
# Step 3.2: Perform morphological operations CLOSE
# -----------------------------------------------------------

closingSize = 3

# Selecting an elliptical kernel
element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * closingSize + 1, 2 * closingSize + 1),
            (closingSize,closingSize))

imageMorphClosed = cv2.morphologyEx(th1,
                                    cv2.MORPH_CLOSE, element2)

closingSize2 = 8

# Selecting an elliptical kernel
element3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * closingSize2 + 1, 2 * closingSize2 + 1),
            (closingSize2,closingSize2))

imageMorphClosed2 = cv2.morphologyEx(th1,
                                    cv2.MORPH_CLOSE, element3)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(imageMorphClosed);
plt.title("MorphoCLose");
plt.subplot(122)
plt.imshow(imageMorphClosed2);
plt.title("MorphoCLose2");
plt.show()

# ------------------------------------------------------------
# Step 3.2: Perform morphological operations CLOSE THEN OPEN
# -----------------------------------------------------------


closingSize2 = 10

# Selecting an elliptical kernel
element3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * closingSize2 + 1, 2 * closingSize2 + 1),
            (closingSize2,closingSize2))

imageMorphClosed = cv2.morphologyEx(th1,
                                    cv2.MORPH_CLOSE, element3)

openingSize2 = 20

# Selecting a elliptical kernel# Selecting a elliptical kernel

element40 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * openingSize2 + 1, 2 * openingSize2 + 1),
            (openingSize2,openingSize2))



imageMorphCloseOpened = cv2.morphologyEx(imageMorphClosed , cv2.MORPH_OPEN,
                        element40,iterations=3)

imageMorphCloseOpened2 = cv2.morphologyEx(imageMorphClosed2 , cv2.MORPH_OPEN,
                        element40,iterations=3)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(imageMorphCloseOpened);
plt.title("MorphoCLoseOpen");
plt.subplot(122)
plt.imshow(imageMorphCloseOpened2);
plt.title("MorphoCLoseOpen");
plt.show()

# ---------------------------------------------------------------------
# Step 3.2: Perform morphological operations CLOSE THEN OPEN INVERTING
# ---------------------------------------------------------------------

imagenin = cv2.bitwise_not(imageMorphCloseOpened)
imagenin2 = cv2.bitwise_not(imageMorphCloseOpened2)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(imagenin);
plt.title("Image inverted");
plt.subplot(122)
plt.imshow(imagenin2);
plt.title("Image inverted 2");
plt.show()



# ---------------------------------------------------------------------
# Step 4.3: Display the detected coins - perfect circunferences
# --------------------------------------------------------------------

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

keypoints = detector.detect(imageMorphCloseOpened)
print("Number of contours found for Just inside = {}".format(len(keypoints)))

#-----------------------------------------------------------------------

image2 = cv2.imread("CoinsB.png")

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
    cv2.circle(image2, (x, y), radius, (0, 255, 0), 12)

plt.figure(figsize=(12, 12))
plt.subplot(111)
plt.imshow(image2[:, :, ::-1]);
plt.title("Just Inside Circles")
plt.show()



# ---------------------------------------------------------------------

# Step 4.4: Perform Connected Component Analysis

# --------------------------------------------------------------------

# ---------------------------------------------------------------------
# Step 4.5: Detect coins using Contour Detection
# --------------------------------------------------------------------

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 200
params.maxThreshold = 255

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(imageMorphCloseOpened)

contours, hierarchy = cv2.findContours(imageMorphCloseOpened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))

cv2.drawContours(image, contours, -1, (0, 255, 0), 12);
for cnt in contours:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"] / M["m00"]))
    y = int(round(M["m01"] / M["m00"]))

    # Mark the center
    cv2.circle(image, (x, y), 10, (255, 0, 0), -1);
    
plt.figure(figsize=(12, 12))
plt.subplot(111)
plt.imshow(image[:, :, ::-1]);
plt.title(" Inside Contours and out side")
plt.show()

# -----------------------------------------------------------------------
# Step 4.5: Detect coins using Contour Detection - remove inner contours
# -----------------------------------------------------------------------

contours2, hierarchy2 = cv2.findContours(th1 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found JUST EXTERNAL = {}".format(len(contours2)))

# Draw all contours
###
cv2.drawContours(im, contours2, -1, (255, 255, 0), 30);
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

# -----------------------------------------------------------------------
# Step 4.5: Detect coins using Contour Detection - PERIMETERS AND AREAS
# -----------------------------------------------------------------------

for index, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
   #areaArray.pop(area)
    print("Contour #{} has area = {} and perimeter = {}".format(index + 1, area, perimeter))
    max_value = np.max(area)

# -----------------------------------------------------------------------
# Step 4.5: Detect coins using Contour Detection - MAX AREA
# -----------------------------------------------------------------------

print("Maximum area = {}".format(area))