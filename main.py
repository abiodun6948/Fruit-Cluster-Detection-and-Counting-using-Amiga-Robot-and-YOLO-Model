from ultralytics import YOLO
import cv2
# load yolov8 model
model = YOLO("models/Abioye_best.pt")
# load video
video_path = "assets/grapev1.mp4"# Specify the path to your video file
cap = cv2.VideoCapture(video_path)  # Open the video file
ret = True  # Initialize ret to True
while ret:
    ret, frame = cap.read()  # Read a frame from the video
        
# detect objects
# Track objects
    results = model.track(frame, persist=True)  # Detect objects in the current frame

# plot results
    frame = results[0].plot()  # Plot the results on the frame
# visualize
    cv2.imshow('frame', frame) 

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break