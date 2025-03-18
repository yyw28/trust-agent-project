from flask import Flask, request, jsonify
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio

app = Flask(__name__)

# Load fine-tuned model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained("path_to_fine_tuned_model")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    data = request.json
    text = data['text']
    # Generate speech using your model (e.g., Tacotron2)
    speech = generate_trustworthy_speech(text)
    return jsonify({"speech": speech})

@app.route('/predict_trustworthiness', methods=['POST'])
def predict_trustworthiness():
    file = request.files['file']
    audio, sample_rate = torchaudio.load(file)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    trustworthiness_score = logits.softmax(dim=-1)[0][predicted_label].item()
    return jsonify({"trustworthiness_score": trustworthiness_score})

def generate_trustworthy_speech(text):
    # Implement speech generation logic using your model (e.g., Tacotron2)
    return "generated_speech"

if __name__ == '__main__':
    app.run()