import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('image.jpg',0)


img = cv.medianBlur(img,5)
ret,th30 = cv.threshold(img,30,255,cv.THRESH_BINARY)
ret,th150 = cv.threshold(img,150,255,cv.THRESH_BINARY)
ret,th200 = cv.threshold(img,200,255,cv.THRESH_BINARY)
th_Mean = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th_Gaussian = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)


plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(th30,cmap = 'gray')
plt.title('Limiar (v=30)'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(th150,cmap = 'gray')

plt.title('Limiar (v=150)'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(th200,cmap = 'gray')
plt.title('Limiar (v=200)'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(th_Mean,cmap = 'gray')
plt.title('Adaptativo da média'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(th_Gaussian,cmap = 'gray')
plt.title('Adaptativo Gaussiano'), plt.xticks([]), plt.yticks([])
plt.show()