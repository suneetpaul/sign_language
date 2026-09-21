import os
import json
import re
import glob
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================

DATASET_FOLDER = r"F:\shared005"

# When True: prints exactly what it WOULD rename, without touching any
# files. Run it this way first, check the printed output looks right,
# then set to False to actually perform the renames.
DRY_RUN = False


# =========================================================
# SANITIZE A TRANSCRIPT STRING INTO A SAFE FILENAME
# =========================================================

def sanitize_filename(text):
    """
    Converts transcript text like "goal" or "how are you?" into a safe,
    consistent filename base: lowercase, spaces -> underscores, and any
    character Windows doesn't allow in filenames stripped out.
    """
    text = text.strip()

    # Remove characters Windows forbids in filenames: < > : " / \ | ? *
    text = re.sub(r'[<>:"/\\|?*]', '', text)

    # Collapse internal whitespace, replace with underscores
    text = re.sub(r'\s+', '_', text)

    return text


# =========================================================
# FIND ALL JSON FILES -> BUILD RENAME PLAN
# =========================================================

json_files = glob.glob(os.path.join(DATASET_FOLDER, "*.json"))

print(f"Found {len(json_files)} JSON files.\n")

# Tracks how many times a sanitized name has been used, so duplicate
# transcripts (two different clips both saying "hello") don't overwrite
# each other -- second occurrence becomes "hello_25" instead of "hello".
name_usage = defaultdict(int)

rename_plan = []  # list of (old_path, new_path) tuples
skipped = []      # list of (num, reason) for anything we couldn't process

for json_path in json_files:

    filename = os.path.basename(json_path)
    num = filename[:-len(".json")]  # e.g. "24.json" -> "24"

    # ---- Read the transcript text ----
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        skipped.append((num, f"could not read/parse JSON: {e}"))
        continue

    transcript_text = data.get("transcript", {}).get("text", "").strip()

    if not transcript_text:
        skipped.append((num, "empty or missing transcript.text"))
        continue

    new_base = sanitize_filename(transcript_text)

    if not new_base:
        skipped.append((num, f"transcript text '{transcript_text}' sanitized to empty string"))
        continue

    # ---- Disambiguate duplicate transcript text ----
    name_usage[new_base] += 1
    if name_usage[new_base] > 1:
        new_base = f"{new_base}_{num}"

    # ---- Find every file belonging to this number ----
    # Matches the bare video file (e.g. "24", no extension) AND anything
    # starting with "num." (e.g. "24.json", "24.pose-dwpose.npz",
    # "24.pose-mediapipe.pose") -- but NOT "240.json" etc, since that
    # doesn't start with "24." exactly.
    group_files = []
    for entry in os.listdir(DATASET_FOLDER):
        full_path = os.path.join(DATASET_FOLDER, entry)
        if not os.path.isfile(full_path):
            continue
        if entry == num or entry.startswith(num + "."):
            group_files.append(entry)

    if not group_files:
        skipped.append((num, "no matching files found on disk"))
        continue

    for entry in group_files:
        if entry == num:
            # Bare video file with no extension on disk
            new_name = new_base
        else:
            # Preserve whatever comes after the number
            # (".json", ".pose-dwpose.npz", ".pose-mediapipe.pose", etc.)
            suffix = entry[len(num):]
            new_name = new_base + suffix

        old_path = os.path.join(DATASET_FOLDER, entry)
        new_path = os.path.join(DATASET_FOLDER, new_name)
        rename_plan.append((old_path, new_path))


# =========================================================
# EXECUTE (OR PREVIEW) THE RENAME PLAN
# =========================================================

print(f"Planned renames: {len(rename_plan)}")
print(f"Skipped groups: {len(skipped)}\n")

for old_path, new_path in rename_plan:
    old_name = os.path.basename(old_path)
    new_name = os.path.basename(new_path)

    if os.path.exists(new_path):
        print(f"  [CONFLICT - SKIPPED] {old_name} -> {new_name} (target already exists)")
        continue

    if DRY_RUN:
        print(f"  [DRY RUN] {old_name} -> {new_name}")
    else:
        os.rename(old_path, new_path)
        print(f"  [RENAMED] {old_name} -> {new_name}")

if skipped:
    print("\n--- Skipped (no rename attempted) ---")
    for num, reason in skipped:
        print(f"  {num}: {reason}")

if DRY_RUN:
    print("\nThis was a DRY RUN -- no files were actually renamed.")
    print("Review the output above, then set DRY_RUN = False at the top of this script to apply it for real.")
else:
    print("\nDone.")