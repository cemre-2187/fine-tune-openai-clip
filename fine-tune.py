import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader
from UTKFace import UTKFaceDataset

# Set the device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocessing method for ViT-B/32
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clipModel")

# Freeze all parameters of the model to avoid updating them during training
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the parameters of the last residual block in the visual transformer to allow fine-tuning
for param in model.visual.transformer.resblocks[-1].parameters():
    param.requires_grad = True

# Define the optimizer (AdamW) and set the learning rate
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)

# Load the UTKFace dataset and preprocess images with the CLIP preprocessing pipeline
dataset = UTKFaceDataset(root_dir="./UTKFace", preprocess=preprocess)

# Create a DataLoader to load the dataset in batches (batch_size=32)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set the number of training epochs
num_epochs = 1

# Set the model in training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    for images, texts in dataloader:
        # Move images and text data to the appropriate device (GPU/CPU)
        images = images.to(device)
        texts = texts.to(device)

        # Forward pass: compute logits for images and text using the CLIP model
        logits_per_image, logits_per_text = model(images, texts)
        
        # Create the ground truth labels (the index of the correct match between image and text)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        # Compute the loss for both image-to-text and text-to-image classification tasks
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
        
        # Average the two losses (image and text)
        loss = (loss_img + loss_txt) / 2

        # Zero the gradients, perform backpropagation, and update the model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")