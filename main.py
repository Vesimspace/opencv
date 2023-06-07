import cv2
from tracker import *
# Tracking to object 
tracker = EuclideanDistTracker()

cam = cv2.VideoCapture("highway.mp4")


object_detector = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=60)

while True:
    ret, frame = cam.read()
    height, width, _ = frame.shape
    lil = frame[340: 720, 500: 800]

    mask = object_detector.apply(lil)
    # Masking the vehicles
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(lil, (x, y), (x + w, y + h), (0, 255, 0), 2)

            detections.append([x, y, w, h])
    
    ids = tracker.update(detections)
    for id in ids:
        x, y, h, w, id_box = id
        # The id number and the rectangle
        cv2.putText(lil, str(id_box), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 20, 20), 2)
        cv2.rectangle(lil, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("lil", lil)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()