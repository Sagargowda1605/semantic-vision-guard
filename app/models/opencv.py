import cv2 as cv 

input= "data/input/Snow_Video.mp4"

cap=cv.VideoCapture(input)

if not cap.isOpened():
    raise FileNotFoundError(f"File is not found at {input} ")

fps=cap.get(cv.CAP_PROP_FPS)
w=cap.get(cv.CAP_PROP_FRAME_WIDTH)
h=cap.get(cv.CAP_PROP_FRAME_HEIGHT)

while True:
    ok,frame=cap.read()

    if not ok:
        break

    cv.imshow("video",frame)

    if cv.waitKey(1) & 0xff == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

