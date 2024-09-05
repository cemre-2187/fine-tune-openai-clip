import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader
from UTKFace import UTKFaceDataset

# CLIP modelini yükleyin
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clipModel")

# Tüm model parametrelerini dondurun
for param in model.parameters():
    param.requires_grad = False

# Son katmanları fine-tune için açın
for param in model.visual.transformer.resblocks[-1].parameters():
    param.requires_grad = True

# Fine-tune edilecek katmanlar için optimizer tanımlayın
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)



# Dataset ve DataLoader'ı ayarlayın
dataset = UTKFaceDataset(root_dir="./UTKFace", preprocess=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Eğitim döngüsü
num_epochs = 10

model.train()
for epoch in range(num_epochs):
    for images, texts in dataloader:
        images = images.to(device)
        texts = texts.to(device)

        # Model tahminleri ve kayıp hesaplama
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        loss_img = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
        loss = (loss_img + loss_txt) / 2

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")