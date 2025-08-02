import os
import torch
import tempfile
import streamlit as st
import whisperx
#import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    whisperx_model = whisperx.load_model("medium", device=device, compute_type="float32")
    processor_punjabi = Wav2Vec2Processor.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")
    model_punjabi = Wav2Vec2ForCTC.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi").to(device)
    return whisperx_model, processor_punjabi, model_punjabi

whisperx_model, processor_punjabi, wav2vec_model_punjabi = load_models()

SUPPORTED_LANGUAGES = {
    "Auto-detect": None,
    "Punjabi (pa)": "pa",
    "Bengali (bn)": "bn",
    "English (en)": "en",
    "Hindi (hi)": "hi",
    "Spanish (es)": "es",
    "French (fr)": "fr",
    "German (de)": "de",
    "Chinese (zh)": "zh",
    "Arabic (ar)": "ar"
}

def video_to_audio(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

def detect_language(audio_path):
    try:
        audio = whisperx.load_audio(audio_path)
        result = whisperx_model.transcribe(audio, batch_size=16)
        return result.get("language")
    except Exception as e:
        st.error(f"Language detection failed: {e}")
        return None

def transcribe_audio(audio_path, lang_code):
    try:
        if lang_code == "pa":
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            inputs = processor_punjabi(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = wav2vec_model_punjabi(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            return processor_punjabi.batch_decode(pred_ids)[0]

        elif lang_code == "bn":
            from banglaspeech2text import Speech2Text
            stt = Speech2Text("base")
            return stt.recognize(audio_path)

        else:
            result = whisperx_model.transcribe(audio_path)
            return result.get("text", "")
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="🎙️ Video Transcriber", layout="centered")
st.title("🎬 Video to Text Transcriber")

uploaded_video = st.file_uploader("Upload a video file:", type=["mp4", "mov", "avi", "mkv", "flv"])

selected_language_label = st.selectbox("Select transcription language:", list(SUPPORTED_LANGUAGES.keys()))
selected_lang_code = SUPPORTED_LANGUAGES[selected_language_label]

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(uploaded_video.read())
        video_path = tmp_vid.name

    st.success("✅ Video uploaded successfully!")
    st.info("Extracting audio...")

    audio_path = video_to_audio(video_path)

    if audio_path:
        st.success("✅ Audio extracted!")

        if selected_lang_code:
            lang_code = selected_lang_code
            st.info(f"Using selected language: `{lang_code}`")
        else:
            st.info("Detecting language automatically...")
            lang_code = detect_language(audio_path)

        if lang_code:
            st.write(f"🈯 Detected/selected language code: `{lang_code}`")
            st.info("Transcribing...")
            transcript = transcribe_audio(audio_path, lang_code)

            if transcript:
                st.success("✅ Transcription complete!")
                st.text_area("📜 Transcript", transcript, height=300)
                st.download_button("⬇️ Download Transcript", transcript, file_name="transcript.txt")
        else:
            st.error("❌ Could not determine language.")
