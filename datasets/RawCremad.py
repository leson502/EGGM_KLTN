import copy
import csv
import os
import pickle
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
import librosa
from torchvision import transforms
import torchaudio
import pandas as pd

THIS_FILE = os.path.abspath(__file__)
from transformers import Wav2Vec2Processor, ViTImageProcessor, ASTFeatureExtractor

MAX_AUDIO_LENGTH = 409600
class CramedDataset(Dataset):

    def __init__(self, audio_path, visual_path, data, train=True):
        
        self.image = []
        self.audio = []
        self.label = []
        self.train = train
        self.audio_length = 256
        self.fps = 1

        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = visual_path
        self.audio_feature_path = audio_path

        self.audio_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        

        for item in data:
            audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
            visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.fps), item[0])

            if os.path.exists(audio_path) and os.path.exists(visual_path):
                self.image.append(visual_path)
                self.audio.append(audio_path)
                self.label.append(class_dict[item[1]])
            else:
                continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        # samples, rate = librosa.load(, sr=22050)
        waveform, sample_rate = torchaudio.load(self.audio[idx])
        
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        waveform = waveform.mean(dim=0)

        spec = self.audio_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                ).input_values[0]
                
        # print(fbank.shape)
        if self.train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=self.fps, replace=False)
        select_index.sort()

        img = Image.open(os.path.join(self.image[idx], image_samples[select_index[0]])).convert('RGB')
        img = transform(img)

        # label
        label = self.label[idx]

        return spec, img, label
    


def load_cremad(root):
    
    train_csv = os.path.join(root, 'cremad/train.csv')
    test_csv = os.path.join(root, 'cremad/test.csv')
    print(train_csv)
    print(test_csv)

    train_df = pd.read_csv(train_csv, header=None)
    test = pd.read_csv(test_csv, header=None)

    audio_path = os.path.join(root, 'cremad/AudioWAV')
    visual_path = os.path.join(root, 'cremad')
    train_dataset = CramedDataset(audio_path,  visual_path, train_df.to_numpy(), True)
    test_dataset = CramedDataset(audio_path,  visual_path, test.to_numpy(), False)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = load_cremad('../data/')
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)
    