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

# --- DANH SÁCH CẢM XÚC CHUẨN HOÁ ---
STANDARD_EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def align_vector(predictions, model_type):
    vec = np.zeros(len(STANDARD_EMOTIONS))
    for pred in predictions:
        label = pred['label'].lower()
        score = pred['score']
        
        if 'joy' in label or 'hap' in label: label = 'happy'
        elif 'ang' in label: label = 'angry'
        elif 'neu' in label: label = 'neutral'
        elif 'sad' in label: label = 'sad'
        elif 'fea' in label: label = 'fear'
        elif 'sur' in label: label = 'surprise'
            
        if label in STANDARD_EMOTIONS:
            idx = STANDARD_EMOTIONS.index(label)
            vec[idx] += score
            
    if np.sum(vec) > 0:
        vec = vec / np.sum(vec)
    return vec

# --- BƯỚC 1: TRÍCH XUẤT ---
def step1_extract_data(video_path):
    audio_path = "temp_audio.wav"
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
    except Exception as e:
        return None, None, f"Lỗi trích xuất audio: {e}"
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 7:
        return None, None, "Video quá ngắn, không đủ khung hình."
        
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
    return audio_path, frames, "Thành công"

# --- BƯỚC 2: GỌI MODEL ---
def step2_get_emotion_vectors(audio_path, frames):
    device = 0 if torch.cuda.is_available() else -1

    # 1. Visual
    vis_pipe = pipeline("image-classification", model="MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12", device=device, top_k=None)
    vis_vectors = [align_vector(vis_pipe(img), "visual") for img in frames]
    visual_vector = np.mean(vis_vectors, axis=0)
    
    # 2. Audio
    audio_pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device, top_k=None)
    audio_vector = align_vector(audio_pipe(audio_path), "audio")

    # 3. Text
    asr_pipe = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)
    transcript = asr_pipe(audio_path)['text']
    
    text_pipe = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier", device=device, top_k=None)
    text_preds = text_pipe(transcript)
    if isinstance(text_preds[0], list): text_preds = text_preds[0]
    text_vector = align_vector(text_preds, "text")

    return visual_vector, audio_vector, text_vector, transcript

# --- BƯỚC 3 & 4: TÍNH TOÁN VÀ KẾT LUẬN ---
def process_video(video_path):
    if not video_path:
        return "Vui lòng tải lên một video!"
        
    yield "⏳ **Đang xử lý:** Đang trích xuất hình ảnh và âm thanh từ video..."
    
    audio_path, frames, msg = step1_extract_data(video_path)
    if not audio_path:
        yield f"❌ **Lỗi:** {msg}"
        return
        
    yield "⏳ **Đang xử lý:** Đang chạy các mô hình AI để phân tích cảm xúc (quá trình này có thể mất vài phút cho lần chạy đầu tiên)..."
    
    visual_vec, audio_vec, text_vec, transcript = step2_get_emotion_vectors(audio_path, frames)
    
    sim_TA = cosine_similarity(text_vec.reshape(1, -1), audio_vec.reshape(1, -1))[0][0]
    sim_TV = cosine_similarity(text_vec.reshape(1, -1), visual_vec.reshape(1, -1))[0][0]
    sim_AV = cosine_similarity(audio_vec.reshape(1, -1), visual_vec.reshape(1, -1))[0][0]
    
    avg_similarity = (sim_TA + sim_TV + sim_AV) / 3
    threshold = 0.5 # Ngưỡng quyết định
    
    # Định dạng kết quả đầu ra
    if avg_similarity < threshold:
        conclusion = "### 🚨 PHÁT HIỆN DEEPFAKE!\n**(Sự mâu thuẫn cảm xúc đa phương thức quá lớn)**"
    else:
        conclusion = "### ✅ VIDEO THẬT (REAL)\n**(Cảm xúc khuôn mặt, giọng nói và ý nghĩa từ ngữ đồng nhất)**"

    result_md = f"""
    {conclusion}
    
    ---
    **🎙️ Lời nói trích xuất được:** "{transcript}"
    
    **📊 CHỈ SỐ LỆCH PHA CẢM XÚC:**
    * 🔹 Khớp Lời nói & Tông giọng (T-A) : **{sim_TA:.2f}**
    * 🔹 Khớp Lời nói & Khuôn mặt (T-V)  : **{sim_TV:.2f}**
    * 🔹 Khớp Tông giọng & Khuôn mặt (A-V): **{sim_AV:.2f}**
    
    ⭐ **ĐỘ NHẤT QUÁN TRUNG BÌNH: {avg_similarity:.2f}** *(Ngưỡng an toàn: > {threshold})*
    """
    
    # Dọn dẹp file tạm
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
        
    yield result_md

# ==========================================
# KHỞI TẠO GIAO DIỆN WEB VỚI GRADIO
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ Demo Trí Tuệ Nhân Tạo Lật Tẩy Deepfake")
    gr.Markdown("Dựa trên nghiên cứu phát hiện Deepfake thông qua **sự lệch pha cảm xúc (Emotion Mismatch)** giữa Khuôn mặt, Tông giọng và Lời nói.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Tải lên video cần kiểm tra (.mp4)")
            submit_btn = gr.Button("🔍 Bắt đầu phân tích", variant="primary")
        
        with gr.Column():
            output_text = gr.Markdown(label="Kết quả phân tích", value="Kết quả sẽ hiển thị ở đây...")
            
    # Xử lý luồng dữ liệu khi bấm nút
    submit_btn.click(fn=process_video, inputs=video_input, outputs=output_text)

# Chạy ứng dụng web
if __name__ == "__main__":
    demo.launch(share=True)
