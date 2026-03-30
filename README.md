# 🕵️‍♂️ DeepFake-Detector-3F (Three-Factor Emotion Mismatch)

A lightweight, multimodal Deepfake detection system based on **Emotion Inconsistency** between Face, Voice, and Text.

##  Core Concept
Unlike traditional detectors that look for pixel artifacts, this project focuses on **high-level semantic mismatch**. A real human naturally aligns their facial expressions, vocal tone, and the meaning of their words. Deepfakes often fail to synchronize these three emotional channels perfectly.

### The Triple-Check System:
1.  **Visual (Face):** Analyzed via `Swin Transformer`.
2.  **Audio (Tone):** Analyzed via `Wav2Vec2`.
3.  **Text (Sentiment):** Speech-to-Text (`Wav2Vec2`) + Sentiment Analysis (`BERT`).

##  Architecture
The system calculates **Cosine Similarity** between the emotion vectors of these three factors. If the average similarity drops below a threshold (e.g., 0.5), the video is flagged as a **Deepfake**.

##  Quick Start (Google Colab)
1. Open [Google Colab](https://colab.research.google.com/).
2. Enable **T4 GPU** (Runtime > Change runtime type).
3. Install dependencies:
   ```bash
   pip install gradio opencv-python moviepy transformers torch scikit-learn
4, copy the code in main.py to the google colab.
5, run 
