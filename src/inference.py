import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os

def get_model(num_classes=102):
    """
    Creates the ResNet50 model architecture as used in training.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    
    # Freeze backbone (though not strictly necessary for inference if we load full state dict, 
    # but good practice to match training struct if we were fine-tuning. 
    # Here we just need the architecture to match keys).
    # Since we are loading a state_dict that presumably contains all keys (backbone + head),
    # we just need to ensure the head dimension matches.
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def load_model(model_path, num_classes=102, device='cpu'):
    """
    Loads the trained model from the checkpoint path.
    """
    model = get_model(num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Load state dict
    # Map to cpu if cuda not available
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle possible key mismatch if model was saved in a wrapper or different way
    # But usually torch.save(model.state_dict()) is straightforward.
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model

def process_image(image):
    """
    Preprocesses the image for the model.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return transform(image).unsqueeze(0) # Add batch dimension

def predict(image, model, device='cpu', topk=3):
    """
    Predicts the class of the image.
    """
    img_tensor = process_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    probs, indices = torch.topk(probabilities, topk)
    
    return probs.cpu().numpy()[0], indices.cpu().numpy()[0]

if __name__ == "__main__":
    # Simple test
    print("Testing model loading...")
    try:
        model_path = "models/best_model.pth" # Assuming running from src/
        if not os.path.exists(model_path):
             model_path = "best_model.pth" # check current dir too
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = load_model(model_path, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
