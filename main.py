from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
import numpy as np
import tensorflow as tf
import librosa
from moviepy.editor import VideoFileClip
import shutil

app = FastAPI(title="ascraa - Audio Deepfake Detection",
              description="Upload a video; the endpoint checks audio for deepfake using audio_classifier.h5",
              version="1.0")

MODEL_PATH = "audio_classifier.h5"   # put your model here
SR = 22050                          # sampling rate for audio extraction
N_MELS = 128
TARGET_FRAMES = 109                 # per architecture diagram
HOP_LENGTH = 512
N_FFT = 2048
THRESHOLD = 0.5                     # prediction threshold to declare deepfake

# Load model once at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    # Keep model as None and raise helpful error on requests if not loaded
    model = None
    load_error = str(e)


def extract_audio_from_video(video_path: str, out_audio_path: str, sr: int = SR):
    """
    Extract audio from a video file and write a WAV file using ffmpeg (via moviepy).
    Requires ffmpeg installed on system and available in PATH.
    """
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise RuntimeError("Uploaded video contains no audio track.")
    # moviepy -> write a WAV file. Use PCM 16-bit for compatibility.
    clip.audio.write_audiofile(out_audio_path, fps=sr, nbytes=2, codec="pcm_s16le")
    clip.close()


def audio_to_log_mel(path_wav: str,
                     sr: int = SR,
                     n_mels: int = N_MELS,
                     n_fft: int = N_FFT,
                     hop_length: int = HOP_LENGTH,
                     target_frames: int = TARGET_FRAMES):
    """
    Load audio and compute log-mel spectrogram, then pad/truncate to (n_mels, target_frames).
    Returns a numpy array shaped (n_mels, target_frames).
    """
    y, sr_loaded = librosa.load(path_wav, sr=sr, mono=True)
    if y.size == 0:
        raise RuntimeError("Loaded audio is empty.")
    # Compute mel-spectrogram (power)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)  # shape (n_mels, T)
    # pad or truncate on the time axis (second dimension) to target_frames
    T = S_db.shape[1]
    if T < target_frames:
        pad_width = target_frames - T
        # pad with minimum dB value (silence)
        min_val = np.min(S_db)
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_val)
    elif T > target_frames:
        # center crop (more robust than taking start)
        start = max(0, (T - target_frames) // 2)
        S_db = S_db[:, start:start + target_frames]
    # Normalization: zero-mean, unit-variance per sample (adjust if your model used a different scheme)
    mean = S_db.mean()
    std = S_db.std() + 1e-9
    S_db = (S_db - mean) / std
    return S_db


@app.post("/predict_audio_deepfake", summary="Predict whether video audio is deepfake")
async def predict_audio_deepfake(file: UploadFile = File(...), threshold: float = THRESHOLD):
    """
    Accepts a video file (multipart/form-data). Extracts audio, computes log-mel spectrogram,
    runs the preloaded Keras model, and returns JSON: { "is_deepfake": bool, "score": float }.
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. Startup load error: {load_error}")

    # Save uploaded video to a temporary file
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_video:
            tmp_video_path = tmp_video.name
            content = await file.read()
            tmp_video.write(content)
        # Extract audio to temp wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            extract_audio_from_video(tmp_video_path, tmp_wav_path, sr=SR)
            # compute spectrogram
            S_db = audio_to_log_mel(tmp_wav_path, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, target_frames=TARGET_FRAMES)
            # reshape to model expected input: (1, 128, 109, 1)
            x = np.expand_dims(S_db, axis=(0, -1)).astype(np.float32)
            # predict
            preds = model.predict(x)
            # If model has single-output sigmoid -> preds shape (1,1) or (1,)
            # If model returns probabilities for classes -> shape (1,2) and we may want prob of 'deepfake' class.
            score = None
            if preds.ndim == 2 and preds.shape[1] == 1:
                score = float(preds[0, 0])
            elif preds.ndim == 2 and preds.shape[1] == 2:
                # assume index 1 is deepfake probability (adjust if training used different ordering)
                score = float(preds[0, 1])
            elif preds.ndim == 1:
                score = float(preds[0])
            else:
                # fallback: take first element
                score = float(preds.flatten()[0])
            is_deepfake = score > float(threshold)
            return JSONResponse(content={"is_deepfake": bool(is_deepfake), "score": float(score)})
        finally:
            # cleanup temp files
            for p in (tmp_video_path, tmp_wav_path):
                try:
                    os.remove(p)
                except Exception:
                    pass

    except Exception as e:
        # ensure temp files removed on error
        try:
            if 'tmp_video_path' in locals() and os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
            if 'tmp_wav_path' in locals() and os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {e}")


if __name__ == "__main__":
    # run with: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
