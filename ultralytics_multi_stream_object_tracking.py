import cv2
from ultralytics import YOLO
import threading

# Define the video files for the trackers
video_file1 = './ultralytics/test.mp4'
video_file2 = './ultralytics/test2.mp4'
video_file3 = 1  # Webcam index

# Load the YOLOv8 models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')
model3 = YOLO('yolov8s.pt')

# Create a stop signal for all threads
stop_event = threading.Event()

# Desired output window size
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

def run_tracker_in_thread(filename, model, file_index):
    video = cv2.VideoCapture(filename)

    while not stop_event.is_set():
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()

        # Resize the output to the same size
        resized_frame = cv2.resize(res_plotted, (WINDOW_WIDTH, WINDOW_HEIGHT))

        cv2.imshow(f"Tracking_Stream_{file_index}", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    video.release()
    cv2.destroyWindow(f"Tracking_Stream_{file_index}")

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)
tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(video_file3, model3, 3), daemon=True)

# Start threads
tracker_thread1.start()
tracker_thread2.start()
tracker_thread3.start()

# Wait for threads to finish
tracker_thread1.join()
tracker_thread2.join()
tracker_thread3.join()

cv2.destroyAllWindows()
