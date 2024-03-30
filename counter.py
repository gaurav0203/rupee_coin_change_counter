import cv2
import numpy as np
import math
from ultralytics import YOLO

model = YOLO("best.pt")
names = model.names
print(names)

calibration_data = np.load('calibration_params.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

cap = cv2.VideoCapture(2)


def preProcessing(frame):
    frame = cv2.undistort(frame, mtx, dist, None)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blurred, 50, 150)

    return canny

def detectCircles(processed_frame, frame, param1, param2,draw = True):
    circles = cv2.HoughCircles(processed_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=param1, param2=param2, minRadius=20, maxRadius=80)
    
    if circles is not None and draw:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    
    return circles, frame 

def drawPrediction(frame, coins_dict, total_value):
    list_num = 1
    for coin, num in coins_dict.items():
        frame = cv2.putText(frame,f'{coin}\'s : {num}', (0, list_num * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        list_num += 1
    
    frame = cv2.putText(frame,f'Total Value : {total_value}', (0, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

# cv2.namedWindow('Circles')
# cv2.createTrackbar('param1', 'Circles', 200, 500, lambda x: None)
# cv2.createTrackbar('param2', 'Circles', 30, 200, lambda x: None)

while True:
    ret, frame = cap.read()

    if ret:
        # param1 = cv2.getTrackbarPos('param1', 'Circles')
        # param2 = cv2.getTrackbarPos('param2', 'Circles')

        param1 = 200
        param2 = 30

        processed_frame = preProcessing(frame)
        circles, frame = detectCircles(processed_frame, frame, param1, param2, draw = True)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if circles is not None:
            total_value = 0
            coins_dict = {1:0,2:0,5:0,10:0,20:0}
            for (x, y, z) in circles:
                startX = max(x - z, 0)
                endX = min(x + z, 640)
                startY = max(y - z, 0)
                endY = min(y + z, 480)
                #print(startX,endX,startY,endY)
                #cv2.imshow("Coins", frame[startY:endY,startX:endX])
                coin_value = 0
                results = model.predict(frame[startY:endY,startX:endX], conf=0.7)
                #for value, template in templates:
                for r in results:
                    coin_value = names[r.probs.top1][-2:]
                    print("Prediction", coin_value)
                #print("Similarity score:", similarity_score)
                total_value += int(coin_value)
                coins_dict[int(coin_value)] += 1
            
            frame = drawPrediction(frame, coins_dict, total_value)
                
            print("Total", total_value)


        if total_value == 18:
            cv2.imwrite("example4.jpg",frame)
            break
        cv2.imshow("Video", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()