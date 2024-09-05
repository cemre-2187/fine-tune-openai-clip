import torch
import clip
from PIL import Image

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", download_root="clipModel")


image = preprocess(Image.open("26_0.jpg")).unsqueeze(0)
text = clip.tokenize(["15 years old female", "26 years old male"])


with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Probabilities:", probs)