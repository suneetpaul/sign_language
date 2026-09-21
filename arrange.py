import cv2
import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FOLDER = r"F:\sign language"

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv", ".flv", ".webm"
)

# Duration folder names (created inside EVERY letter folder)
DURATION_FOLDERS = [
    "0-5_seconds",
    "6-10_seconds",
    "10-30_seconds",
    "30-60_seconds",
    "above_60_seconds",
]


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

    return frame_count / fps


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
# PICK FOLDER NAME FOR A DURATION
# ==========================================

def pick_folder_name(duration):

    if duration <= 5:
        return "0-5_seconds"

    if duration <= 10:
        return "6-10_seconds"

    if duration <= 30:
        return "10-30_seconds"

    if duration <= 60:
        return "30-60_seconds"

    return "above_60_seconds"


# ==========================================
# SORT ONE LETTER FOLDER
# ==========================================

def sort_one_folder(base_folder):

    # Create the 5 duration folders inside this letter folder
    for name in DURATION_FOLDERS:
        os.makedirs(os.path.join(base_folder, name), exist_ok=True)

    counts = {name: 0 for name in DURATION_FOLDERS}
    failed = 0

    for root, dirs, files in os.walk(base_folder):

        # Don't walk into the duration folders themselves
        dirs[:] = [d for d in dirs if d not in DURATION_FOLDERS]

        for file in files:

            if not file.lower().endswith(VIDEO_EXTENSIONS):
                continue

            video_path = os.path.join(root, file)

            duration = get_video_duration(video_path)

            if duration is None:
                failed += 1
                continue

            folder_name = pick_folder_name(duration)

            target_folder = os.path.join(base_folder, folder_name)

            destination = get_unique_path(target_folder, file)

            try:
                shutil.move(video_path, destination)
                counts[folder_name] += 1

            except Exception as e:
                print("Move failed:", video_path, e)
                failed += 1

    return counts, failed


# ==========================================
# RUN FOR EVERY LETTER FOLDER
# ==========================================

total = {name: 0 for name in DURATION_FOLDERS}
total_failed = 0

letter_folders = sorted(
    d for d in os.listdir(INPUT_FOLDER)
    if os.path.isdir(os.path.join(INPUT_FOLDER, d))
    and d not in DURATION_FOLDERS
)

for letter in letter_folders:

    base_folder = os.path.join(INPUT_FOLDER, letter)

    counts, failed = sort_one_folder(base_folder)

    total_failed += failed

    for name in DURATION_FOLDERS:
        total[name] += counts[name]

    print(
        f"{letter}: "
        + ", ".join(f"{n}={counts[n]}" for n in DURATION_FOLDERS)
        + f", failed={failed}"
    )


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== SORTING COMPLETE ==========")

print("Folders processed:", len(letter_folders))
print("0-5 seconds:", total["0-5_seconds"])
print("Above 5 to 10 seconds:", total["6-10_seconds"])
print("Above 10 to 30 seconds:", total["10-30_seconds"])
print("Above 30 to 60 seconds:", total["30-60_seconds"])
print("Above 60 seconds:", total["above_60_seconds"])
print("Failed:", total_failed)

print("Input folder:", INPUT_FOLDER)