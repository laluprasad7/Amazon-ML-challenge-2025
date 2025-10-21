import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
import joblib
from config import cache_dir, batch_size

def load_or_compute_features(data, images, prefix, cache_dir=cache_dir, batch_size=batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Feature extraction using device: {device}")

    text_cache = f'{cache_dir}/{prefix}_text_feats.pkl'
    img_cache = f'{cache_dir}/{prefix}_img_feats.pkl'

    # Text features: DistilBERT for speed
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = BertModel.from_pretrained('distilbert-base-uncased').to(device)
    bert_model.eval()

    if os.path.exists(text_cache):
        text_feats = joblib.load(text_cache)
        print(f"Loaded cached text features from {text_cache}")
    else:
        def get_text_features(texts):
            features = []
            for i in tqdm(range(0, len(texts), batch_size), desc=f"{prefix} Text Features"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # CLS token
            return np.vstack(features)
        text_feats = get_text_features(data['clean_text'].tolist())
        joblib.dump(text_feats, text_cache)
        print(f"Saved text features to {text_cache}")

    # Image features: MobileNetV3 for speed
    resnet = models.mobilenet_v3_small(pretrained=True).to(device)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(device)
    resnet.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(img_cache):
        img_feats = joblib.load(img_cache)
        print(f"Loaded cached image features from {img_cache}")
    else:
        def get_image_features(images):
            features = []
            for i in tqdm(range(0, len(images), batch_size), desc=f"{prefix} Image Features"):
                batch = [transform(img) for img in images[i:i+batch_size]]
                batch = torch.stack(batch).to(device)
                with torch.no_grad():
                    feats = resnet(batch).squeeze()
                features.append(feats.cpu().numpy())
            return np.vstack(features)
        img_feats = get_image_features(images)
        joblib.dump(img_feats, img_cache)
        print(f"Saved image features to {img_cache}")

    return np.hstack([text_feats, img_feats, data['ipq'].values.reshape(-1, 1)])