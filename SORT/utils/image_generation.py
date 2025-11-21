

import os 
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pathlib import Path 
import cv2
import numpy as np
from SORT.simulation_od_module.simulation_od_module import SimulationOD
import pandas as pd 
import random




ID_COLORS = {}


def get_id_color(obj_id):
    """Return a unique BGR color for each ID."""
    if obj_id not in ID_COLORS:
        ID_COLORS[obj_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return ID_COLORS[obj_id]


def create_image(image_path:Path,time:int, object_detection_output:pd.DataFrame, save_path:Path) -> None:
    """
    Create an image with bounding boxes from object detection output.

    Args:
        image_path (Path): Path to the original image.
        object_detection_output (np.ndarray): Array containing object detection results.
        save_path (Path): Path to save the generated image.
    """

    if not save_path.exists():
        os.makedirs(save_path, exist_ok=True)
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'. Check the path and file permissions.")

    # Normalize and validate save_path: if empty or missing extension, pick a sensible default
    save_path = Path(save_path) / (str(time)+ ".png")


    # Draw bounding boxes
    
    for idx in range(len(object_detection_output)):
        detection = object_detection_output.iloc[idx]
        obj_id = detection['id']
        x = int(detection['x'])
        y = int(detection['y'])
        w = int(detection['w'])
        h = int(detection['h'])
        
        
        ## add text of id 

        label = str(int(obj_id))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = get_id_color(int(obj_id))

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # measure text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # # text position: 5 px above the box
        text_x = x
        text_y = y - 5

        cv2.rectangle(
            image,
            (text_x, text_y - text_h - 5),          # top-left of background
            (text_x + text_w + 10, text_y + 5),     # bottom-right of background
            color,
            -1  # filled
        )

        # draw the label text
        cv2.putText(
            image,
            label,
            (text_x + 5, text_y - 15 + text_h),   # text inside the background rectangle
            font,
            font_scale,
            (255, 255, 255),  # white text
            thickness
        )

    # Save the generated image
    success = cv2.imwrite(str(save_path), image)
    if not success:
        raise RuntimeError(f"OpenCV failed to write image to '{save_path}'. Check that the file extension is supported.")

def create_video(path_to_save_images:Path, video_name:str="output_video.avi", fps:int=30) -> None:
    """
    Create a video from a sequence of images stored in a directory.

    Args:
        path_to_save_images (Path): Directory containing the images.
        video_name (str): Name of the output video file.
        fps (int): Frames per second for the output video.
    """
    images = [img for img in os.listdir(path_to_save_images) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split(".")[0]))  # Sort images numerically based on filename

    if not images:
        raise ValueError(f"No images found in directory '{path_to_save_images}'.")

    # Read the first image to get the dimensions
    first_image_path = path_to_save_images / images[0]
    frame = cv2.imread(str(first_image_path))
    height, width, layers = frame.shape

    video_path = path_to_save_images / video_name
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for image_name in images:
        image_path = path_to_save_images / image_name
        frame = cv2.imread(str(image_path))
        video.write(frame)

    video.release()

    
if __name__ == "__main__":

    path_to_scenario_folder= Path(r"C:\Projects\datasets\MOT17\MOT17_dataset\train\MOT17-02-FRCNN")
    simulation_od  = SimulationOD(path_to_scenario_folder = path_to_scenario_folder)
    simulation_od.reset()
    N = len(simulation_od)

    path_to_save_images = Path("images_and_video\\SORT")

    for i in range(N):
        state = simulation_od[i]
        object_detection_mudule_output = state['od_gt']
        create_image(
            image_path=Path(state['frame_img_path']),
            time = i,
            object_detection_output=object_detection_mudule_output,
            save_path = path_to_save_images)
    
    create_video(path_to_save_images = path_to_save_images)


