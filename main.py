import cv2
import numpy as np

'''
img = cv2.imread('img/cartao.jpg')

#cv2.imshow('imagem', img)
#cv2.waitKey(0)

width, height = 300,500
pts1 = np.float32([[168,198],[31,419],[576,337],[487,614]])
pts2 = np.float32([[0,height],[width,height],[0,0],[width,0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_out = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow('imagem', img_out)
cv2.waitKey(0)
'''

kernel = np.ones((5, 5), np.uint8)


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 1)
    img = cv2.Canny(img, 200, 200)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>50:
            print(area)
            #cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            aprox = cv2.approxPolyDP(cnt, 0.02*peri, True)
            '''
            print(len(aprox))
            objCor = len(aprox)
            x, y, w, h = cv2.boundingRect(aprox)
            '''
            if area > maxArea and len(aprox) == 4:
                biggest = aprox
                maxArea = area
    cv2.drawContours(imgContours, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder(points):
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew

def getWarp(img, biggest):
    biggest = reorder(biggest)
    width, height = 640, 480
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_out = cv2.warpPerspective(img, matrix, (width, height))
    return img_out



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 130)

while True:
    success, img = cap.read()
    cv2.resize(img, (640, 480))
    imgContours = img.copy()
    imgThres = preprocessing(img)
    biggest = getContours(imgThres)
    print(biggest)
    if biggest.size != 0:
        img_warp = getWarp(img, biggest)
    else:
        img_warp = imgThres
    
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 0)
    img = cv2.Canny(img, 150, 200)
    img = cv2.dilate(img, kernel=np.ones((5,5),np.uint8), iterations=1)
    img = cv2.erode(img, kernel=np.ones((5,5),np.uint8), iterations=1)
    img = cv2.resize(img, (1000, 500))
    img = img[0:500, 500:1000]
    '''
    cv2.imshow('video', img_warp)
    if cv2.waitKey(1) % 0xFF ==ord('q'):
        break

