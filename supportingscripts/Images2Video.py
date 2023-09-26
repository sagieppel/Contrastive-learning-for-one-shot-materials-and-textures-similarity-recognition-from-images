import cv2
import os

# Directory containing all images
img_folder_path = r"/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/VideoMatSim/BreadAndCheese/1mp4to3dir/"
images = [img for img in os.listdir(img_folder_path) if img.endswith(".jpg")]

# Sort the images by frame number
images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))

frame = cv2.imread(os.path.join(img_folder_path, images[0]))
height, width, layers = frame.shape

# Define the codec using VideoWriter_fourcc and create VideoWriter object
# You can use 'XVID' codec for .avi output format, or 'MP4V' for .mp4
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi', fourcc, 30, (width, height)) # The '1' denotes FPS. Modify as needed.

for ff,image in enumerate(images):
    print(ff)
    video_frame = cv2.imread(os.path.join(img_folder_path, image))
    out.write(video_frame)

out.release()

print("The video has been successfully created!")