import cv2
import numpy as np
import duidie as dd

#轮廓检测
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #寻找轮廓函数,第二个参数为检测方法
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)   #计算边线弧度周长,封闭的为真
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)  #jiaodiandezhi
            #几边形 几 = len(approx)
            #绘制边界框，并得到位置,为封闭图形添加边框
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 2)




img = cv2.imread("picture/1.jpg")
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)  #模糊处理
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)

#堆叠
imgStack = dd.stackImages(0.6, ([img, imgGray, imgContour],
                                [imgCanny,imgBlur,img]))

# cv2.imshow("yuantu", img)
# cv2.imshow("original", imgGray)
# cv2.imshow("blur", imgBlur)
cv2.imshow("duidie",imgStack)
cv2.waitKey(0)

