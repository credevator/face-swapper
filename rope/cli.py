import os
import cv2
import numpy as np
from PIL import Image
import argparse
import torch
from onnxruntime import InferenceSession
import torchvision.transforms as transforms

from VideoManager import VideoManager
from Models import Models  # Assuming you have a Models class for face detection and recognition

def main(input_video_path, input_image_path, output_video_path):
    # Initialize models and video manager
    models = Models()
    vm = VideoManager(models)

    # Load input image and video
    vm.load_target_image(input_image_path)
    vm.load_target_video(input_video_path)

    # Simulate finding faces and assigning found faces (normally done by GUI)
    vm.find_faces()

    # Assuming only one face to swap from input image, assigning found faces manually
    if vm.target_faces:
        first_face = vm.target_faces[0]  # Use the first detected face for swapping
        first_face["ButtonState"] = True
        vm.assign_found_faces([first_face])

    # Process the video frames with swapping
    output_temp_file = output_video_path + "_temp.avi"
    frame_width = int(vm.capture.get(3))
    frame_height = int(vm.capture.get(4))
    fps = vm.fps

    size = (frame_width, frame_height)
    out = cv2.VideoWriter(output_temp_file, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    while vm.capture.isOpened():
        ret, frame = vm.capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        swapped_frame = vm.swap_video(rgb_frame, vm.current_frame, use_markers=False)  # Perform face swap
        
        swapped_frame_bgr = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
        out.write(swapped_frame_bgr)
        
        vm.current_frame += 1

    vm.capture.release()
    out.release()

    # Add audio to the final output video
    orig_file = input_video_path
    final_file = output_video_path
    print("Adding audio to the final output video...")
    args = ["ffmpeg",
            '-hide_banner',
            '-loglevel', 'error',
            "-i", output_temp_file,
            "-i", orig_file,
            "-c", "copy",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            final_file]
    subprocess.run(args)
    os.remove(output_temp_file)
    print(f'Video saved as: {final_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Swap CLI")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("input_image", help="Path to the input image file")
    parser.add_argument("output_video", help="Path to save the output video file")
    
    args = parser.parse_args()
    main(args.input_video, args.input_image, args.output_video)
