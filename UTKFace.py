import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, preprocess, transform=None):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.image_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        image = self.preprocess(image)

        filename = self.image_files[idx]
        age, gender, race = filename.split("_")[:3]
        
        person_gender="female"
        if gender == "0":
           person_gender="male"
            
        
        text = f"A {person_gender} who is {age} years old"
        # print(text)
        return image, clip.tokenize([text])[0]