import cv2
import os
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================

INPUT_FOLDER = r"F:\video"

OUTPUT_FOLDER = r"F:\video\frame"

# Frames per second to extract
TARGET_FPS = 15

# CNN input size
IMAGE_SIZE = (224, 224)


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def process_video(video_path, output_folder):

    # Skip videos already processed -- safe to re-run this script after
    # adding new videos without redoing everything.
    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        return

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR:", video_path)
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        print("Invalid FPS:", video_path)
        cap.release()
        return

    frame_interval = max(
        1,
        round(original_fps / TARGET_FPS)
    )

    frame_number = 0
    saved_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % frame_interval == 0:

            # Direct resize (no cropping) -- note this stretches/squishes
            # non-square source frames into a square, since aspect ratio
            # isn't preserved. Removed per request.
            frame = cv2.resize(
                frame,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            filename = f"{saved_frames:05d}.jpg"

            output_path = os.path.join(
                output_folder,
                filename
            )

            cv2.imwrite(
                output_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            saved_frames += 1

        frame_number += 1

    cap.release()

    print(
        f"{os.path.basename(video_path)} "
        f"-> {saved_frames} frames"
    )


# =========================================================
# PROCESS ALL VIDEOS -- PRESERVING FOLDER STRUCTURE
# =========================================================
#
# If your word-level videos are grouped the same way as sentence-level
# (one folder per word, containing multiple videos of different
# signers), this mirrors that grouping in the output -- one input
# folder with N videos becomes one output folder with N subfolders
# of frames, e.g.:
#
#   Videos_Word_Level/ABUSE/abuse1.mp4, abuse2.mp4, ...
#     -> SignBridge_Processed/word/ABUSE/abuse1/00000.jpg, ...
#     -> SignBridge_Processed/word/ABUSE/abuse2/00000.jpg, ...
#
# If a video sits directly inside INPUT_FOLDER with no grouping
# subfolder, it still works -- falls back to one folder per video.

video_files = []

for root, dirs, files in os.walk(INPUT_FOLDER):

    for file in files:

        if file.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):

            video_files.append(
                os.path.join(root, file)
            )


print("Videos found:", len(video_files))


for video_path in tqdm(video_files):

    filename = os.path.basename(video_path)

    video_name = os.path.splitext(filename)[0]
    video_name = video_name.replace(" ", "_")

    # Relative path from INPUT_FOLDER to the video's containing folder --
    # this is what preserves the grouping (e.g. "ABUSE").
    video_dir = os.path.dirname(video_path)
    rel_group_path = os.path.relpath(video_dir, INPUT_FOLDER)
    rel_group_path = rel_group_path.replace(" ", "_")

    if rel_group_path == ".":
        # Video was directly inside INPUT_FOLDER with no grouping subfolder
        output_folder = os.path.join(OUTPUT_FOLDER, video_name)
    else:
        output_folder = os.path.join(OUTPUT_FOLDER, rel_group_path, video_name)

    process_video(
        video_path,
        output_folder
    )


print("\nDONE!")
print("Processed frames:", OUTPUT_FOLDER)