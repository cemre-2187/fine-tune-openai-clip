import torch
import clip
from PIL import Image
import numpy as np
from classificationTest import classificationTextData

# Load the CLIP model (ViT-B/32) and preprocessing method
model, preprocess = clip.load("ViT-B/32", download_root="clipModel")

# Load and preprocess the image. Then, add a batch dimension using unsqueeze(0)
image = preprocess(Image.open("test.jpg")).unsqueeze(0)

# Tokenize the classification text data (convert the text into input suitable for CLIP)
text = clip.tokenize(classificationTextData)

# Disable gradient computation since we're just performing inference
with torch.no_grad():
    # Extract features from the image using the CLIP model
    image_features = model.encode_image(image)
    
    # Extract features from the text data using the CLIP model
    text_features = model.encode_text(text)

    # Get the similarity scores between the image and the text using the CLIP model
    logits_per_image, logits_per_text = model(image, text)
    
    # Apply softmax to the similarity scores to get the probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Uncomment the following line to print the probabilities
# print("Probabilities:", probs)

# Convert the probabilities to a numpy array
probsValues = np.array(probs)

# Find the maximum probability value
max_value = np.max(probsValues)

# Get the index of the class with the highest probability
max_index = np.argmax(probsValues)

# Print the highest probability and the corresponding class index
print(f"Highest probability: {max_value}, Index: {max_index}")

# Print the class with the highest probability
print(f"Class with the highest probability: {classificationTextData[max_index]}")