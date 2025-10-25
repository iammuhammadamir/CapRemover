# trim_prefix_cv.py
import sys
import cv2

if len(sys.argv) != 3:
    print("Usage: python trim_prefix_cv.py <vid_name> <seconds>")
    sys.exit(1)

vid_name = sys.argv[1]
seconds = float(sys.argv[2])

cap = cv2.VideoCapture(vid_name)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(seconds * fps)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'trimmed_{vid_name}', fourcc, fps, (width, height))

count = 0
while count < frame_count:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    count += 1

cap.release()
out.release()
