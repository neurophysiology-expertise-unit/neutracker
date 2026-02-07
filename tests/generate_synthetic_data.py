
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

def generate_synthetic_eye(filename, n_frames=100, width=320, height=240):
    """
    Generates a video of a moving white circle (pseudo-pupil) on black background.
    """
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height), isColor=False)
    
    print(f"Generating {n_frames} frames to {filename}...")
    
    center_x, center_y = width // 2, height // 2
    radius = 30
    
    for i in tqdm(range(n_frames)):
        # Create black image
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Move the pupil
        offset_x = int(50 * np.sin(2 * np.pi * i / n_frames))
        offset_y = int(20 * np.cos(2 * np.pi * i / n_frames))
        
        # Draw pupil (white circle)
        cv2.circle(frame, (center_x + offset_x, center_y + offset_y), radius, (255), -1)
        
        # Draw "Corneal Reflection" (small static white dot)
        cv2.circle(frame, (center_x + 10, center_y - 10), 3, (200), -1)
        
        out.write(frame)
        
    out.release()
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Output video file (e.g. test_video.avi)')
    args = parser.parse_args()
    
    generate_synthetic_eye(args.output)
    
if __name__ == '__main__':
    main()
