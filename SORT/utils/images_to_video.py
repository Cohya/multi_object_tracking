import cv2
import os 

def images_to_video(path_to_save_images, path_to_save_video='output.mp4', fps=30):
    # Get all image files in the folder, sorted
    images = [img for img in os.listdir(path_to_save_images) 
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # make sure the order is correct

    if len(images) == 0:
        print("No images found in folder!")
        return

    # Read the first image to get the size
    first_image_path = os.path.join(path_to_save_images, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    out = cv2.VideoWriter(path_to_save_video, fourcc, fps, (width, height))

    # Add all images to the video
    for img_name in images:
        img_path = os.path.join(path_to_save_images, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: Could not read {img_name}, skipping.")
            continue

        # Resize if needed (ensure same size)
        frame = cv2.resize(frame, (width, height))

        out.write(frame)

    out.release()
    print(f"Video saved to {path_to_save_video}")