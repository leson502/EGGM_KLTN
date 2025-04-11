import random

import torch
import random
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import os
import numpy as np

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
    def __init__(self, feature_path, data_path):
        super(FoodDataset, self).__init__()
        df = pd.read_csv(data_path)
        self.data = []
        self.visual = []
        self.text = []
        for item in df.iterrows():
            index, row = item
            name = row.iloc[0].replace('.jpg', '')
            label = row.iloc[2]

            visual_path = join(feature_path, "image", name + '.npy')
            text_path = join(feature_path, "text", name + '.npy')

            if not os.path.exists(visual_path) or not os.path.exists(text_path):
                print(f"File not found: {visual_path} or {text_path}")
                # print(f"File not found: {name}")
                continue
            self.visual.append(np.load(visual_path))
            self.text.append(np.load(text_path))
            
            self.data.append((visual_path, text_path, label))
        
        self.label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAME)}
        self.idx_to_label = {idx: label for idx, label in enumerate(CLASS_NAME)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        visual_path, text_path, label = self.data[index]
        visual = torch.from_numpy(self.visual[index]).float().squeeze(0)
        text = torch.from_numpy(self.text[index]).float()
        label = self.label_to_idx[label]
        return visual, text, label


if __name__ == "__main__":
    dataset = FoodDataset('/home/sonlt/workspace/markdocdown/EGGM/data/food/features', '/home/sonlt/workspace/markdocdown/EGGM/data/food/texts/test_titles.csv')
    print(len(dataset))
    for i in range(5):
        visual, text, label = dataset[i]
        print(visual.shape, text.shape, label)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for visual, text, label in loader:
        print(visual.shape, text.shape, label)
        break

            
        
        
