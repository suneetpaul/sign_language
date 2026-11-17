# Generated from: SignBright_Video_and_Live_Webcam_Recognition.ipynb
# Converted at: 2026-11-11T12:16:06.910Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # SignBright — Video & Live Webcam Recognition
# 
# This notebook loads your **continued-training** word and sentence Transformer checkpoints and lets you test:
# 1. An uploaded video
# 2. A live webcam in Google Colab (browser camera permission required)
# 
# **Important:** the S3 screenshot shows the model checkpoints, but not the three files needed to turn class IDs into labels and apply the same training normalization. Upload these too:
# - `word_labels.json`
# - `sentence_labels.json`
# - `normalization.npz`
# 
# They are from `/opt/signbright/output2/model/` in the training setup you shared. The continued checkpoints are the `*_continued_best.pth` files shown in your S3 bucket.
# 
# The notebook matches the training pipeline: 64 frames × 1,662 features, MediaPipe Hands + Face Mesh + Pose, and the saved `mean`/`std` normalization. Word mode uses one center frame (your word training data used single images); sentence mode samples a 64-frame sequence.


# 1) Install the required package
# Run once. In Google Colab, PyTorch and OpenCV are usually already available.
%pip -q install mediapipe==0.10.21

# 2) Upload your 5 required files
from google.colab import files
uploaded = files.upload()

required = [
    "word_transformer_continued_best.pth",
    "sentence_transformer_continued_best.pth",
    "word_labels.json",
    "sentence_labels.json",
    "normalization.npz",
]
missing = [name for name in required if name not in uploaded]
if missing:
    raise FileNotFoundError(
        "Please upload these missing files and rerun this cell:\n" + "\n".join(missing)
    )
print("All model and metadata files uploaded.")

# 3) Imports and settings
import os, json, base64, time, collections
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from IPython.display import display, Javascript, HTML, clear_output
from google.colab import output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 64
FEATURES = 1662
HAND_FEATURES = 126
FACE_FEATURES = 1404
POSE_FEATURES = 132

with open("word_labels.json", "r", encoding="utf-8") as f:
    word_to_id = {str(k): int(v) for k, v in json.load(f).items()}
with open("sentence_labels.json", "r", encoding="utf-8") as f:
    sentence_to_id = {str(k): int(v) for k, v in json.load(f).items()}

norm = np.load("normalization.npz")
mean = norm["mean"].astype(np.float32)
std = norm["std"].astype(np.float32)
std[std < 1e-6] = 1.0
assert mean.shape == (FEATURES,) and std.shape == (FEATURES,), (
    f"Normalization must have {FEATURES} values; got {mean.shape} and {std.shape}"
)

print("Device:", DEVICE)
print("Word classes:", len(word_to_id))
print("Sentence classes:", len(sentence_to_id))
print("Feature shape expected: (64, 1662)")

# 4) Exact SignBright Transformer architecture and checkpoint loading
MAX_FRAMES = SEQ_LEN

class SignTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.input_layer = nn.Linear(FEATURES, 256)
        self.positional_embedding = nn.Parameter(torch.zeros(1, MAX_FRAMES, 256))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.positional_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.classifier(x)

def load_sign_model(path, label_map, model_name):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"{model_name} checkpoint does not contain the expected 'model' weights.")
    if checkpoint.get("features", FEATURES) != FEATURES:
        raise ValueError(f"{model_name} checkpoint feature count is not 1662.")
    if checkpoint.get("max_frames", SEQ_LEN) != SEQ_LEN:
        raise ValueError(f"{model_name} checkpoint frame count is not 64.")
    model = SignTransformer(len(label_map)).to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    print(f"Loaded {model_name} model.")
    return model

word_model = load_sign_model(
    "word_transformer_continued_best.pth", word_to_id, "word"
)
sentence_model = load_sign_model(
    "sentence_transformer_continued_best.pth", sentence_to_id, "sentence"
)

word_id_to_label = {v: k for k, v in word_to_id.items()}
sentence_id_to_label = {v: k for k, v in sentence_to_id.items()}

# 5) MediaPipe feature extraction — same landmark order and normalization as training
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=True, model_complexity=1,
    enable_segmentation=False, min_detection_confidence=0.5
)

def normalize_hand(hand):
    points = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
    points = points - points[0]  # wrist origin
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points.flatten()

def normalize_face(face):
    points = np.array([[p.x, p.y, p.z] for p in face.landmark], dtype=np.float32)
    points = points - points[1]  # nose landmark as center
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points.flatten()

def normalize_pose(pose_landmarks):
    points = np.array(
        [[p.x, p.y, p.z, p.visibility] for p in pose_landmarks.landmark],
        dtype=np.float32
    )
    center = (points[23, :3] + points[24, :3]) / 2.0
    points[:, :3] -= center
    scale = np.max(np.linalg.norm(points[:, :3], axis=1))
    if scale > 0:
        points[:, :3] /= scale
    return points.flatten()

def extract_frame_features(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)
    pose_result = pose.process(rgb)

    hand_features = np.zeros(HAND_FEATURES, dtype=np.float32)
    face_features = np.zeros(FACE_FEATURES, dtype=np.float32)
    pose_features = np.zeros(POSE_FEATURES, dtype=np.float32)

    if hand_result.multi_hand_landmarks:
        left, right = None, None
        for i, hand in enumerate(hand_result.multi_hand_landmarks):
            data = normalize_hand(hand)
            side = hand_result.multi_handedness[i].classification[0].label
            if side == "Left":
                left = data
            else:
                right = data
        if left is not None:
            hand_features[:63] = left
        if right is not None:
            hand_features[63:126] = right

    if face_result.multi_face_landmarks:
        face_features = normalize_face(face_result.multi_face_landmarks[0])
    if pose_result.pose_landmarks:
        pose_features = normalize_pose(pose_result.pose_landmarks)

    result = np.concatenate([hand_features, face_features, pose_features]).astype(np.float32)
    if result.shape[0] != FEATURES:
        raise ValueError(f"Feature extraction returned {result.shape[0]}, expected {FEATURES}")
    return result

def sample_video_frames(video_path, count=SEQ_LEN):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open this video. Try MP4, MOV, or AVI.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read video frames.")
    indices = np.linspace(0, total - 1, min(count, total)).astype(int)
    frames = []
    current_index = 0
    wanted = set(indices.tolist())
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current_index in wanted:
            frames.append(frame)
        current_index += 1
    cap.release()
    return frames

def prepare_features(frame_features, mode):
    frame_features = np.asarray(frame_features, dtype=np.float32)
    if mode == "word":
        # Word training samples were still images: use the center frame and zero-pad.
        if len(frame_features) == 0:
            raise ValueError("No frames were extracted.")
        one = frame_features[len(frame_features) // 2]
        sequence = np.zeros((SEQ_LEN, FEATURES), dtype=np.float32)
        sequence[0] = one
    else:
        if len(frame_features) == 0:
            raise ValueError("No frames were extracted.")
        if len(frame_features) > SEQ_LEN:
            ids = np.linspace(0, len(frame_features) - 1, SEQ_LEN).astype(int)
            frame_features = frame_features[ids]
        sequence = np.zeros((SEQ_LEN, FEATURES), dtype=np.float32)
        sequence[:len(frame_features)] = frame_features

    # Same training-time standardization and clipping, including padded frames.
    sequence = (sequence - mean) / std
    sequence = np.clip(sequence, -5.0, 5.0)
    return torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)

def predict(mode, frame_features, top_k=5):
    if mode == "word":
        model, id_to_label = word_model, word_id_to_label
    else:
        model, id_to_label = sentence_model, sentence_id_to_label
    x = prepare_features(frame_features, mode)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        values, indices = torch.topk(probs, k=min(top_k, len(id_to_label)))
    return [(id_to_label[int(i)], float(p) * 100) for p, i in zip(values.cpu(), indices.cpu())]

print("MediaPipe and prediction functions are ready.")

# ## 6) Test an uploaded video
# 
# Choose **word** for a single sign/word clip or **sentence** for a continuous sequence. The sentence model samples 64 frames across the video.


from google.colab import files
import tempfile

print("Upload one video file (MP4 / MOV / AVI).")
video_upload = files.upload()
if not video_upload:
    raise RuntimeError("No video uploaded.")
video_name = next(iter(video_upload))
with open(video_name, "wb") as f:
    f.write(video_upload[video_name])

MODE = "word"  # change to "sentence" for sentence recognition

frames = sample_video_frames(video_name, SEQ_LEN)
print("Frames sampled:", len(frames))
frame_features = [extract_frame_features(frame) for frame in frames]
predictions = predict(MODE, frame_features, top_k=5)

print(f"\n{MODE.upper()} RECOGNITION — TOP PREDICTIONS")
for rank, (label, confidence) in enumerate(predictions, 1):
    print(f"{rank}. {label}: {confidence:.2f}%")

# Preview one sampled frame
display(HTML("<h4>Video frame preview</h4>"))
preview = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_BGR2RGB)
display(__import__("PIL").Image.fromarray(preview))

# ## 7) Live webcam recognition
# 
# Run the cell below, allow browser camera access, then sign in front of the camera. Choose `word` or `sentence` in the setting. The live session runs for 60 seconds; interrupt the cell to stop sooner.
# 
# - **Word mode:** predicts from each current frame, consistent with the single-image word training samples.
# - **Sentence mode:** builds a rolling 64-frame clip and predicts the sequence.


# Set the live recognition mode:
LIVE_MODE = "word"   # "word" or "sentence"
LIVE_SECONDS = 60
LIVE_FPS = 8

# Browser webcam bridge (works in Google Colab; click Allow when prompted).
display(HTML("""
<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
  <button id="sbStart" style="padding:9px 16px">Start camera</button>
  <button id="sbStop" style="padding:9px 16px">Stop camera</button>
  <span id="sbStatus">Camera not started</span>
</div>
<video id="sbVideo" autoplay playsinline width="480"
       style="margin-top:10px;max-width:100%;border-radius:12px;background:#111"></video>
<canvas id="sbCanvas" width="640" height="480" style="display:none"></canvas>
"""))
display(Javascript("""
(async () => {
  const video = document.getElementById('sbVideo');
  const status = document.getElementById('sbStatus');
  window.signbrightStop = false;
  document.getElementById('sbStart').onclick = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
      video.srcObject = stream;
      await video.play();
      status.textContent = 'Camera running — sign now';
      window.signbrightStop = false;
    } catch (e) { status.textContent = 'Camera error: ' + e.message; }
  };
  document.getElementById('sbStop').onclick = () => {
    window.signbrightStop = true;
    if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
    status.textContent = 'Camera stopped';
  };
  window.signbrightGetFrame = () => {
    if (window.signbrightStop || !video.srcObject || video.readyState < 2) return null;
    const canvas = document.getElementById('sbCanvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.75);
  };
})();
"""))

# Click "Start camera" above before running this cell.
from collections import deque

print("Waiting for webcam frames...")
time.sleep(2)
recent_frames = deque(maxlen=SEQ_LEN)
last_inference = 0
started = time.time()
interval = 1.0 / LIVE_FPS
last_result = None

try:
    while time.time() - started < LIVE_SECONDS:
        if output.eval_js("window.signbrightStop === true"):
            print("Webcam stopped.")
            break

        data_url = output.eval_js("window.signbrightGetFrame()")
        if not data_url:
            time.sleep(0.1)
            continue

        encoded = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(encoded, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        recent_frames.append(frame)
        now = time.time()

        # Word mode can predict from the current frame; sentence mode waits for a full clip.
        ready = (LIVE_MODE == "word") or (len(recent_frames) == SEQ_LEN)
        if ready and now - last_inference >= 1.0:
            if LIVE_MODE == "word":
                chosen = [frame]
            else:
                chosen = list(recent_frames)
            feats = [extract_frame_features(f) for f in chosen]
            last_result = predict(LIVE_MODE, feats, top_k=3)
            last_inference = now

            clear_output(wait=True)
            print(f"LIVE {LIVE_MODE.upper()} RECOGNITION")
            print(f"Elapsed: {int(now-started)}s")
            for rank, (label, confidence) in enumerate(last_result, 1):
                print(f"{rank}. {label}: {confidence:.2f}%")
            display(HTML("<small>To stop: click Stop camera above or interrupt this cell.</small>"))

        time.sleep(interval)

except KeyboardInterrupt:
    print("Live recognition interrupted.")

finally:
    try:
        output.eval_js("""
        (() => {
          const v = document.getElementById('sbVideo');
          if (v && v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
          window.signbrightStop = true;
        })()
        """)
    except Exception:
        pass
    print("Webcam session ended.")

# ## Notes on interpreting results
# 
# - The percentages are **model confidence scores**, not guaranteed real-world accuracy.
# - Test with signs that are represented in the training labels and use similar camera framing, lighting, and signing style.
# - If predictions look unrelated, first confirm that the two continued checkpoints, both label JSON files, and `normalization.npz` all came from the same training run.
# - Your screenshot's S3 list contains the two continued best checkpoints, checkpoints/last files, and config. It does **not** show the two label JSON files or `normalization.npz`; those are also required for correct inference.