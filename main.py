import cv2
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import warnings
import os

warnings.filterwarnings("ignore")

# --- STANDARDIZED EMOTION LIST ---
STANDARD_EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def align_vector(predictions, model_type):
    """Maps various model outputs to a consistent 6-dimensional emotion vector."""
    vec = np.zeros(len(STANDARD_EMOTIONS))
    for pred in predictions:
        label = pred['label'].lower()
        score = pred['score']
        
        # Mapping variations to standard labels
        if 'joy' in label or 'hap' in label: label = 'happy'
        elif 'ang' in label: label = 'angry'
        elif 'neu' in label: label = 'neutral'
        elif 'sad' in label: label = 'sad'
        elif 'fea' in label: label = 'fear'
        elif 'sur' in label: label = 'surprise'
            
        if label in STANDARD_EMOTIONS:
            idx = STANDARD_EMOTIONS.index(label)
            vec[idx] += score
            
    # Normalize the vector so the sum is 1
    if np.sum(vec) > 0:
        vec = vec / np.sum(vec)
    return vec

# --- STEP 1: EXTRACTION ---
def step1_extract_data(video_path):
    audio_path = "temp_audio.wav"
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
    except Exception as e:
        return None, None, f"Audio extraction error: {e}"
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 7:
        return None, None, "Video is too short, not enough frames."
        
    frame_indices = np.linspace(0, total_frames - 1, 7, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((224, 224))
            frames.append(img)
    cap.release()
    return audio_path, frames, "Success"

# --- STEP 2: MODEL INVOCATION ---
def step2_get_emotion_vectors(audio_path, frames):
    device = 0 if torch.cuda.is_available() else -1

    # 1. Visual (Face)
    vis_pipe = pipeline("image-classification", model="MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12", device=device, top_k=None)
    vis_vectors = [align_vector(vis_pipe(img), "visual") for img in frames]
    visual_vector = np.mean(vis_vectors, axis=0)
    
    # 2. Audio (Tone)
    audio_pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device, top_k=None)
    audio_vector = align_vector(audio_pipe(audio_path), "audio")

    # 3. Text (Sentiment)
    asr_pipe = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)
    transcript = asr_pipe(audio_path)['text']
    
    text_pipe = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier", device=device, top_k=None)
    text_preds = text_pipe(transcript)
    if isinstance(text_preds[0], list): text_preds = text_preds[0]
    text_vector = align_vector(text_preds, "text")

    return visual_vector, audio_vector, text_vector, transcript

# --- STEP 3 & 4: CALCULATION AND CONCLUSION ---
def process_video(video_path):
    if not video_path:
        return "Please upload a video file!"
        
    yield "⏳ **Processing:** Extracting images and audio from the video..."
    
    audio_path, frames, msg = step1_extract_data(video_path)
    if not audio_path:
        yield f"❌ **Error:** {msg}"
        return
        
    yield "⏳ **Processing:** Running AI models to analyze emotions (this might take a few minutes for the first run)..."
    
    visual_vec, audio_vec, text_vec, transcript = step2_get_emotion_vectors(audio_path, frames)
    
    sim_TA = cosine_similarity(text_vec.reshape(1, -1), audio_vec.reshape(1, -1))[0][0]
    sim_TV = cosine_similarity(text_vec.reshape(1, -1), visual_vec.reshape(1, -1))[0][0]
    sim_AV = cosine_similarity(audio_vec.reshape(1, -1), visual_vec.reshape(1, -1))[0][0]
    
    avg_similarity = (sim_TA + sim_TV + sim_AV) / 3
    threshold = 0.5 # Decision Threshold
    
    # Output formatting
    if avg_similarity < threshold:
        conclusion = "### 🚨 DEEPFAKE DETECTED!\n**(Significant multimodal emotion mismatch)**"
    else:
        conclusion = "### ✅ REAL VIDEO\n**(Facial expressions, vocal tone, and text sentiment are consistent)**"

    result_md = f"""
    {conclusion}
    
    ---
    **🎙️ Extracted Transcript:** "{transcript}"
    
    **📊 EMOTION MISMATCH METRICS:**
    * 🔹 Text & Audio Tone (T-A)   : **{sim_TA:.2f}**
    * 🔹 Text & Visual Face (T-V)  : **{sim_TV:.2f}**
    * 🔹 Audio Tone & Visual Face (A-V): **{sim_AV:.2f}**
    
    ⭐ **AVERAGE CONSISTENCY: {avg_similarity:.2f}** *(Safety threshold: > {threshold})*
    """
    
    # Cleanup temporary files
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
        
    yield result_md

# ==========================================
# GRADIO WEB INTERFACE INITIALIZATION
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ AI Deepfake Detection Demo")
    gr.Markdown("Based on detecting Deepfakes via **Emotion Mismatch** across Facial Expressions, Vocal Tones, and Speech Content.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload a video to test (.mp4)")
            submit_btn = gr.Button("🔍 Start Analysis", variant="primary")
        
        with gr.Column():
            output_text = gr.Markdown(label="Analysis Result", value="The result will appear here...")
            
    # Process data stream on button click
    submit_btn.click(fn=process_video, inputs=video_input, outputs=output_text)

# Run the web app
if __name__ == "__main__":
    demo.launch(share=True)
