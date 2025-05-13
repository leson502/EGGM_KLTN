import random

import torch
import random
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import os
import glob
import numpy as np
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor

CLASS_NAME = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
              'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli',
              'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry',
              'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
              'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
              'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
              'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
              'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
              'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
              'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
              'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
              'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
              'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
              'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
              'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
TEXT_MAX_LENGTH = 50
MIN_FREQ = 3
NUMBER_OF_SAMPLES_PER_CLASS = None


class FoodDataset(Dataset):
    def __init__(self, feature_path, data_path, tokenizer_path, processor_path):
        super(FoodDataset, self).__init__()
        df = pd.read_csv(data_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.processor = ViTImageProcessor.from_pretrained(processor_path)
        self.visual = []
        self.text = []
        self.label = []
        for item in df.iterrows():
            index, row = item
            name = row.iloc[0].replace('.jpg', '')
            label = row.iloc[2]

            visual_path = join(feature_path,"images", "all_images", name + '.jpg')

            if os.path.exists(visual_path) == False:
                print(f"File not found: {visual_path}")
                continue
            self.visual.append(visual_path)
            self.text.append(row.iloc[1])
            self.label.append(label)
            
        
        self.label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAME)}
        self.idx_to_label = {idx: label for idx, label in enumerate(CLASS_NAME)}
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        visual_path, text, label = self.visual[index], self.text[index], self.label[index]
        visual = Image.open(visual_path).convert('RGB')
        label = self.label_to_idx[label]
        visual = self.processor(images=visual, return_tensors="pt")['pixel_values'][0]
        text = self.tokenizer.encode_plus(text, max_length=TEXT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt').input_ids[0]
        return visual, text, label


if __name__ == "__main__":
    dataset = FoodDataset('/home/sonlt/workspace/markdocdown/EGGM/data/food/', 
                          '/home/sonlt/workspace/markdocdown/EGGM/data/food/texts/test_titles.csv', 
                          "google-bert/bert-base-uncased",
                          "google/vit-base-patch16-224")
    
    for i in range(5):
        visual, text, label = dataset[i]
        print(visual.shape, text.shape, label)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, batch in enumerate(loader):
        visual, text, label = batch
        print(visual.shape, text.shape, label)
        break
        

            
        
        
