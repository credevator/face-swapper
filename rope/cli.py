import argparse
import os
import cv2
import numpy as np
import torch
import subprocess
from VideoManager import VideoManager
from Models import Models  # Ensure this import works correctly for your models

def main(input_video_path, input_image_path, output_video_path):
    # Initialize models and video manager
    models = Models()
    vm = VideoManager(models)

    # Load input image and video
    vm.load_target_image(input_image_path)
    vm.load_target_video(input_video_path)

    # Detect face in the input image
    input_image = cv2.imread(input_image_path)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_embedding = vm.find_faces_in_image(input_image_rgb)
    vm.found_faces = [{'Embedding': input_embedding}]

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
