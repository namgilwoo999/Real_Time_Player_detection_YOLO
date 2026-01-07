import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Can't open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Input video FPS: {fps}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[INFO] Total frames read: {len(frames)}")
    return frames, fps


def save_video(frames, output_path, fps):
    if not frames:
        raise ValueError("No frames to save")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[INFO] Video saved to {output_path} with FPS: {fps}")