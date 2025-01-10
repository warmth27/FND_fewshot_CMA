import os.path
from collections import defaultdict
import tqdm, random
from torch.utils.data import Dataset, Subset
import csv, torch, json
import numpy as np
import pandas as pd
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device, jit=False)

class DGM4_Dataset(Dataset):
    def __init__(self, data_path, mode="binary"):
        self.root = "D:/Datasets/"
        labels = ['face_swap&text_attribute', 'text_swap', 'face_attribute', 
                  'face_attribute&text_swap', 'face_swap', 'orig', 
                  'text_attribute', 'face_swap&text_swap', 'face_attribute&text_attribute']
        self.class2id = {label: i for i, label in enumerate(labels)}

        with open(data_path, 'r') as inf:
            self.data = json.load(inf)

        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        image = os.path.join(self.root, line['image'])
        label =line['fake_cls']
        text = line['text']
        if self.mode == "binary":
            if label in ['face_swap&text_attribute', 'text_swap', 'face_attribute',
                            'face_attribute&text_swap', 'face_swap',
                            'text_attribute', 'face_swap&text_swap', 'face_attribute&text_attribute']:
                label = 1
            else:
                label = 0
        else:
            label = self.class2id[label]

        # txt = clip.tokenize(text, truncate=True).squeeze().to(device)

        # img = preprocess(Image.open(image).convert("RGB")).to(device)
        # label = torch.as_tensor(label).to(device, torch.long)   

        return image, text, label

class DGM4_Dataset_Test(Dataset):
    def __init__(self, data_path, mode="binary"):
        self.root = "D:/Datasets/"
        labels = ['face_swap&text_attribute', 'text_swap', 'face_attribute', 
                  'face_attribute&text_swap', 'face_swap', 'orig', 
                  'text_attribute', 'face_swap&text_swap', 'face_attribute&text_attribute']
        self.class2id = {label: i for i, label in enumerate(labels)}

        with open(data_path, 'r') as inf:
            self.data = json.load(inf)

        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        image = os.path.join(self.root, line['image'])
        label =line['fake_cls']
        text = line['text']
        if self.mode == "binary":
            if label in ['face_swap&text_attribute', 'text_swap', 'face_attribute',
                            'face_attribute&text_swap', 'face_swap',
                            'text_attribute', 'face_swap&text_swap', 'face_attribute&text_attribute']:
                label = 1
            else:
                label = 0
        else:
            label = self.class2id[label]

        txt = clip.tokenize(text, truncate=True).squeeze().to(device)

        img = preprocess(Image.open(image).convert("RGB")).to(device)
        label = torch.as_tensor(label).to(device, torch.long)   

        return txt, img, label


class FewShotSampler_DGM4():
    def __init__(self, dataset, few_shot_per_class, seed):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class
        self.seed = seed
    def get_train_val_datasets(self):
        indices_per_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[label].append(idx)

        train_indices = []
        val_indices = []

        for label, indices in indices_per_class.items():
            random.Random(self.seed).shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])
            val_indices.extend(indices[self.few_shot_per_class:])

        train_dataset = Subset(self.dataset, train_indices)
        # val_dataset = Subset(self.dataset, val_indices)

        encoded_train_dataset = []
        for image, text, label in train_dataset:
            txt = clip.tokenize(text, truncate=True).squeeze().to(device)
            img = preprocess(Image.open(image).convert("RGB")).to(device)
            label = torch.as_tensor(label).to(device, torch.long) 
            encoded_train_dataset.append((txt, img, label))
            

        return encoded_train_dataset
    

# test_path = "D:/Datasets/DGM4/metadata/test.json"

# data = DGM4_Dataset_Test(test_path, "binary")

# print(data[0])

# train_dataset = FewShotSampler_DGM4(data, 2, 1024).get_train_val_datasets()

# for i in train_dataset:
#     print(i)