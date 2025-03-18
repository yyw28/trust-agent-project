from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
import datasets
import torch
import torchaudio

# Load your dataset
dataset = datasets.load_dataset("path_to_your_dataset", split="train")

# Preprocess the dataset
def preprocess_function(examples):
    audio = examples["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    inputs["labels"] = torch.tensor(examples["label"])
    return inputs

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
dataset = dataset.map(preprocess_function, remove_columns=["audio"])

# Load the pretrained model
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()