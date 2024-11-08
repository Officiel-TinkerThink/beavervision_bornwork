# test_video_creator.py
import numpy as np
import cv2

def create_sample_video():
    # Create a 5-second video at 30 fps
    out = cv2.VideoWriter('sample.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         30, 
                         (640,480))
    
    # Create frames with a face-like shape
    for _ in range(150):  # 5 seconds * 30 fps
        frame = np.zeros((480,640,3), np.uint8)
        # Draw face circle
        cv2.circle(frame, (320,240), 100, (255,255,255), -1)
        # Draw eyes
        cv2.circle(frame, (280,200), 20, (0,0,0), -1)
        cv2.circle(frame, (360,200), 20, (0,0,0), -1)
        # Draw mouth
        cv2.ellipse(frame, (320,280), (40,20), 0, 0, 180, (0,0,0), -1)
        out.write(frame)
    
    out.release()

if __name__ == "__main__":
    create_sample_video()