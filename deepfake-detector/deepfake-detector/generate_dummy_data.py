import cv2
import numpy as np
import os

def create_video(filename, is_fake):
    width, height = 256, 256
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(60): # 2 seconds of video
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Center of the face/circle
        cx = width // 2 + int(np.sin(i * 0.1) * 20)
        cy = height // 2 + int(np.cos(i * 0.1) * 20)
        
        # Draw a face (circle)
        cv2.circle(frame, (cx, cy), 50, (200, 200, 200), -1)
        
        if is_fake:
            # Add glitch/noise artifacts
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            # Occasionally shift the face unnaturally
            if i % 10 == 0:
                cv2.circle(frame, (cx + 30, cy - 30), 50, (150, 150, 150), -1)
                
        out.write(frame)
        
    out.release()

def main():
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    
    print("Generating REAL videos...")
    for i in range(15):
        create_video(f'dataset/real/real_{i}.avi', is_fake=False)
        
    print("Generating FAKE videos...")
    for i in range(15):
        create_video(f'dataset/fake/fake_{i}.avi', is_fake=True)
        
    print("Done generating dummy data.")

if __name__ == "__main__":
    main()
