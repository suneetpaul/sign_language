
import cv2
import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FOLDER = r"F:\video"

OUTPUT_FOLDER = INPUT_FOLDER

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv", ".flv", ".webm"
)

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

FOLDER_1 = os.path.join(OUTPUT_FOLDER, "0-5_seconds")
FOLDER_2 = os.path.join(OUTPUT_FOLDER, "6-10_seconds")
FOLDER_3 = os.path.join(OUTPUT_FOLDER, "10-30_seconds")
FOLDER_4 = os.path.join(OUTPUT_FOLDER, "30-60_seconds")
FOLDER_5 = os.path.join(OUTPUT_FOLDER, "above_60_seconds")

os.makedirs(FOLDER_1, exist_ok=True)
os.makedirs(FOLDER_2, exist_ok=True)
os.makedirs(FOLDER_3, exist_ok=True)
os.makedirs(FOLDER_4, exist_ok=True)
os.makedirs(FOLDER_5, exist_ok=True)

# ==========================================
# GET VIDEO DURATION
# ==========================================

def get_video_duration(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open:", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0 or frame_count <= 0:
        print("Invalid video:", video_path)
        return None

    duration = frame_count / fps

    return duration


# ==========================================
# PREVENT FILE NAME DUPLICATES
# ==========================================

def get_unique_path(folder, filename):

    name, extension = os.path.splitext(filename)

    destination = os.path.join(folder, filename)

    counter = 1

    while os.path.exists(destination):

        new_name = f"{name}_{counter}{extension}"

        destination = os.path.join(folder, new_name)

        counter += 1

    return destination


# ==========================================
# SCAN AND SORT VIDEOS
# ==========================================

count_0_5 = 0
count_6_10 = 0
count_10_30 = 0
count_30_60 = 0
count_above_60 = 0
count_failed = 0

for root, dirs, files in os.walk(INPUT_FOLDER):

    # Skip the output folder if it is inside input
    dirs[:] = [
        d for d in dirs
        if os.path.abspath(os.path.join(root, d))
        != os.path.abspath(OUTPUT_FOLDER)
    ]

    for file in files:

        if not file.lower().endswith(VIDEO_EXTENSIONS):
            continue

        video_path = os.path.join(root, file)

        duration = get_video_duration(video_path)

        if duration is None:
            count_failed += 1
            continue

        # Choose folder based on duration

        if duration <= 5:

            target_folder = FOLDER_1
            count_0_5 += 1

        elif duration <= 10:

            target_folder = FOLDER_2
            count_6_10 += 1

        elif duration <= 30:

            target_folder = FOLDER_3
            count_10_30 += 1

        elif duration <= 60:

            target_folder = FOLDER_4
            count_30_60 += 1

        else:

            target_folder = FOLDER_5
            count_above_60 += 1

        # Avoid overwriting files with the same name

        destination = get_unique_path(
            target_folder,
            file
        )

        try:

            shutil.move(video_path, destination)

            '''print(
                f"{duration:.2f} sec | "
                f"{file} -> {os.path.basename(target_folder)}"
            )'''

        except Exception as e:

            print("Move failed:", video_path, e)
            count_failed += 1


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== SORTING COMPLETE ==========")

print("0-5 seconds:", count_0_5)
print("Above 5 to 10 seconds:", count_6_10)
print("Above 10 to 30 seconds:", count_10_30)
print("Above 30 to 60 seconds:", count_30_60)
print("Above 60 seconds:", count_above_60)
print("Failed:", count_failed)

print("Output folder:", OUTPUT_FOLDER)