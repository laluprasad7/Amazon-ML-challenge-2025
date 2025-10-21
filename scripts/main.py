import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from data_loader import load_data
from image_downloader import download_all_images
from text_preprocessor import preprocess_text_data
from image_preprocessor import preprocess_images
from feature_extractor import load_or_compute_features
from model_trainer import train_model
from predictor import predict_and_save

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df, test_df = load_data()

    # Download images
    download_all_images(train_df, test_df)

    # Preprocess text
    train_df = preprocess_text_data(train_df)
    test_df = preprocess_text_data(test_df)

    # Preprocess images
    train_images = preprocess_images(train_df, 'images/train', "Train Images")
    test_images = preprocess_images(test_df, 'images/test', "Test Images")

    # Split train into train/val
    train_df, val_df, train_images, val_images = train_test_split(
        train_df, train_images, test_size=0.2, random_state=42
    )

    # Extract features
    train_feats = load_or_compute_features(train_df, train_images, 'train')
    val_feats = load_or_compute_features(val_df, val_images, 'val')
    test_feats = load_or_compute_features(test_df, test_images, 'test')

    # Train model
    train_prices = np.log1p(train_df['price'].values)
    val_prices = np.log1p(val_df['price'].values)
    model = train_model(train_feats, train_prices, val_feats, val_prices)

    # Predict and save
    predict_and_save(model, test_feats, test_df)

if __name__ == "__main__":
    main()