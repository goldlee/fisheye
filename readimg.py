import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

c = 0
while ret:
    cv2.imwrite(str(c) + '.jpg', frame)
    cv2.imshow('aa', frame)
    k = cv2.waitKey(2)
    if k == ord('s'):
        cv2.imwrite(str(c) + '.jpg', frame)
        c += 1
    if k == ord('q'):
        break;
    ret, frame = cap.read()
