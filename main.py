# trust_agent_pipeline.py

import os
import joblib
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import opensmile
import gradio as gr
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# HuBERT-based Trust Regression Model
class HuBERTTrustRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.reg_head = nn.Sequential(
            nn.Linear(self.hubert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_values, attention_mask=None):
        with torch.no_grad():
            outputs = self.hubert(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        return self.reg_head(hidden_states).squeeze()


class HuBERTTrustModel:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = HuBERTTrustRegressor().to(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.model.eval()

    def predict(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # convert to mono
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        inputs = self.feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            score = self.model(input_values).cpu().item()
        return round(min(max(score, 0.0), 1.0), 2)


# BERT-based Trust Regression Model
class BERTTrustModel:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.reg_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(device)
        self.bert.eval()
        self.reg_head.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            score = self.reg_head(cls_embedding).squeeze().cpu().item()
        return round(min(max(score, 0.0), 1.0), 2)



# Whisper-based Transcription Model
class WhisperWrapper:
    def __init__(self, model_name="openai/whisper-base", device="cpu"):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    def transcribe(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_features"].to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_values)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]



# GPT-based Rewriter using OpenAI or local LLM
class GPTRewriter:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.generator = pipeline("text2text-generation", model=model_name)

    def rewrite(self, text, trust_score):
        prompt = f"The following speech has a trustworthiness score of {trust_score:.2f}. Rewrite it to sound more trustworthy, confident, and clear. Also provide a feedback message to the speaker on how to improve vocal delivery (e.g., pitch, speed, tone).\n\nSpeech:\n{text}\n\nRewritten + Feedback:"
        rewritten = self.generator(prompt, max_new_tokens=200)[0]['generated_text']
        return rewritten.strip()


class TrustAgent:
    def __init__(self, acoustic_model, text_model, asr_model, rewriter):
        self.acoustic_model = acoustic_model
        self.text_model = text_model
        self.transcriber = asr_model
        self.rewriter = rewriter

    def analyze(self, audio_path):
        transcript = self.transcriber.transcribe(audio_path)
        acoustic_score = self.acoustic_model.predict(audio_path)
        text_score = self.text_model.predict(transcript)
        original_score = round((acoustic_score + text_score) / 2, 2)

        rewritten_text = self.rewriter.rewrite(transcript, original_score)
        revised_score = self.text_model.predict(rewritten_text)

        return transcript, original_score, rewritten_text, revised_score


if __name__ == "__main__":
    acoustic_model = HuBERTTrustModel()
    text_model = BERTTrustModel()
    transcriber = WhisperWrapper()
    rewriter = GPTRewriter()

    agent = TrustAgent(acoustic_model, text_model, transcriber, rewriter)

    def interface_fn(audio):
        transcript, original_score, rewritten_text, revised_score = agent.analyze(audio)
        return transcript, original_score, rewritten_text, revised_score

    demo = gr.Interface(
        fn=interface_fn,
        inputs=gr.Audio(source="upload", type="filepath"),
        outputs=[
            gr.Textbox(label="Original Transcript"),
            gr.Number(label="Original Trust Score"),
            gr.Textbox(label="Rewritten Speech + Feedback"),
            gr.Number(label="Rewritten Trust Score"),
        ],
        title="Trust-Aware Speech Analyzer with Feedback",
        description="Upload speech to evaluate and improve its trustworthiness using acoustic and lexical features. Suggestions are provided to enhance delivery and language."
    )
    demo.launch()

