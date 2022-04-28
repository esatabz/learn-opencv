import cv2
import duidie as dd

faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
img = cv2.imread('resources/lena.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# cv2.imshow("gray", imgGray)
# cv2.imshow("img", img)
#堆叠
imgStack = dd.stackImages(0.6, ([img, imgGray]))
cv2.imshow("stack", imgStack)
cv2.waitKey(0)






