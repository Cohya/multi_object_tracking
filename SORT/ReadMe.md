# Simple Online and Realtime Tracking (SORT)

In this small project we are going to use MOT17 dataset. (https://motchallenge.net/data/MOT17/)

## Sort Building Blocks 

 1) Track representation 
    - S(t) $=$ (x,y,dx,dy,w,h)- stands for center of the bounding box velocity and width and height of the detected object each frame. 

 2) Prediction Step 
    - Before matching detections, each track (in the memory) predicts its S(t+1) using tracking algorithm. In this small project we are going to use **Kalman Filter** for it.


 3) Data Association  
    - Match predicted tracks to new detections
    - Compute cost fucntion using Intersection over union (IOU)
    - In this project we are going to solve the assignment optimally using **Hungarian algorithm**.

4) Track Management
    - Create new tracks for unmatched detections
    - Delete old tracks if they haven't been matched for N frames 
    - Update matched tracks with **KF** measurement update 

## Peogram Architecture 

![](./drawio_files/architecture.drawio.svg)
