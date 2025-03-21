import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from transformers import Wav2Vec2Model, ViTModel
import numpy as np
import librosa
import cv2
from dtw import accelerated_dtw
from pydub import AudioSegment

class VideoFeatureExtractor(nn.Module):
    def __init__(self, num_classes=768):
        super(VideoFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Identity()  
        self.fc = nn.Linear(1280, num_classes)  
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.efficientnet(x)
        features = features.view(batch_size, seq_length, -1)
        features = torch.mean(features, dim=1)  
        return self.fc(features)


class AudioFeatureExtractor(nn.Module):
    def __init__(self, num_classes=768):
        super(AudioFeatureExtractor, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.fc = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)
    
    def forward(self, x):
        outputs = self.wav2vec2(x)
        features = outputs.last_hidden_state.mean(dim=1) 
        return self.fc(features)

class DeepfakeClassifier(nn.Module):
    def __init__(self, visual_dim=768, audio_dim=768, hidden_dim=256, output_dim=2):
        super(DeepfakeClassifier, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, visual, audio):
        visual = visual.unsqueeze(0) 
        audio = audio.unsqueeze(0) 
        attn_output, _ = self.cross_attention(visual, audio, audio)
        fusion = torch.cat((attn_output.squeeze(0), audio.squeeze(0)), dim=1)
        return self.fc(fusion)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()
    return torch.stack(frames).unsqueeze(0)  


def process_audio(video_path):
    audio_path = "temp_audio.wav"
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export(audio_path, format="wav")
    
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = torch.tensor(audio).unsqueeze(0).float()
    return audio

if __name__ == "__main__":
    video_path = "sample.mp4"
    
    try:
        video_frames = process_video(video_path)
        audio_features = process_audio(video_path)
    except ValueError as e:
        print(f"Error: {e}")
        exit()
    
    video_model = VideoFeatureExtractor()
    audio_model = AudioFeatureExtractor()
    classifier = DeepfakeClassifier()
    
    video_model.eval()
    audio_model.eval()
    classifier.eval()
    
    with torch.no_grad():
        video_features = video_model(video_frames)
        audio_features = audio_model(audio_features)
    
    prediction = classifier(video_features, audio_features)
    real_probability = prediction[0, 0]
    if(real_probability>0.5): prediction = "real"
    else: prediction = "fake"