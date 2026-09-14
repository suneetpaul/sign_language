# sign_language_project

# Dataset Preparation Scripts

This repo contains two scripts used to prepare a sign language video dataset for training: one that converts raw videos into frame images, and one that renames dataset files based on their transcript text.

## 1. `vido_to_image2.py` — Video → Frames Extractor

Walks through a folder of sign language videos and converts each one into a sequence of resized JPEG frames, ready for use as CNN input.

### What it does
- Recursively scans `INPUT_FOLDER` for video files (`.mp4`, `.avi`, `.mov`, `.mkv`).
- For each video, extracts frames and saves them as `00000.jpg`, `00001.jpg`, ... inside a matching output folder.
- Preserves the input folder structure, so if videos are grouped by word/sentence (e.g. `are_you_free_today/video1.mp4`), the output mirrors that grouping (e.g. `are_you_free_today/video1/00000.jpg`).
- Resizes each frame with **aspect-ratio-preserving resize + center crop** (`resize_and_crop`) instead of stretching it, so hand shapes/proportions aren't distorted.
- Skips any video whose output folder already has frames in it, so the script is safe to re-run after adding new videos.

### Key settings (top of file)
| Variable | Purpose |
|---|---|
| `INPUT_FOLDER` | Path to the raw videos |
| `OUTPUT_FOLDER` | Where extracted frames are written |
| `TARGET_FPS` | Frames per second to extract (only used if `EXTRACT_ALL_FRAMES = False`) |
| `EXTRACT_ALL_FRAMES` | If `True`, keeps every frame at the video's native FPS (max smoothness, larger output). If `False`, downsamples to `TARGET_FPS`. |
| `IMAGE_SIZE` | Output frame size, default `(224, 224)` for CNN input |

> **Note:** If you change `TARGET_FPS`, `EXTRACT_ALL_FRAMES`, or `IMAGE_SIZE` after already running the script once, delete the relevant output folders first — the skip-if-exists check won't regenerate them automatically.

### Dependencies
```
opencv-python (cv2)
```

### Usage
Set `INPUT_FOLDER` and `OUTPUT_FOLDER` at the top of the script, then run:
```bash
python vido_to_image2.py
```

---

## 2. `rename_address.py` — Transcript-Based File Renamer

Renames dataset files (video, JSON transcript, pose files, etc.) from numeric IDs to human-readable names based on the transcript text stored in each JSON file.

### What it does
- Scans `DATASET_FOLDER` for all `*.json` files (e.g. `24.json`).
- Reads `transcript.text` from each JSON to get the sign's meaning (e.g. `"how are you"`).
- Sanitizes that text into a filesystem-safe name (lowercase, spaces → underscores, illegal characters stripped).
- Finds every file on disk associated with that numeric ID — the bare video file, the `.json`, and any related files like `.pose-dwpose.npz` or `.pose-mediapipe.pose` — and renames them all to share the new base name while keeping their original suffix.
- Handles **duplicate transcripts**: if two different clips say the same thing (e.g. two `"hello"` videos), the second one becomes `hello_<id>` instead of overwriting the first.
- Skips and logs anything it can't process: unreadable/invalid JSON, empty transcript text, or missing files.

### Safety features
- **`DRY_RUN` flag**: when `True`, prints exactly what would be renamed without touching any files. Run once with `DRY_RUN = True` to verify the plan, then set to `False` to apply it.
- **Conflict detection**: if a target filename already exists, that rename is skipped and reported instead of overwriting.

### Key settings (top of file)
| Variable | Purpose |
|---|---|
| `DATASET_FOLDER` | Folder containing the JSON transcripts + associated dataset files |
| `DRY_RUN` | `True` = preview only, `False` = actually rename files |

### Dependencies
Standard library only: `os`, `json`, `re`, `glob`, `collections`.

### Usage
```bash
python rename_address.py
```
Review the printed dry-run output first, then set `DRY_RUN = False` and re-run to apply the renames.

---
