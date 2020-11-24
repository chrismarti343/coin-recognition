import cv2

img = cv2.imread("CoinsA.png")

cv2.imshow("Image",img)
k = cv2.waitKey(2000)
print(k)