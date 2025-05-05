import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch

CLASSES = ['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon']
IMAGE_SIZE = 224

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image) -> np.ndarray:
    input_tensor = val_test_transforms(image)
    input_numpy = input_tensor.unsqueeze(0).numpy()
    return input_numpy

def postprocess_output(output_logits: np.ndarray) -> dict:
    output_tensor = torch.from_numpy(output_logits)
    probabilities = torch.softmax(output_tensor, dim=1)
    max_prob, predicted_class_index_tensor = torch.max(probabilities, dim=1)
    predicted_class_index = predicted_class_index_tensor.item()
    confidence_score = max_prob.item()
    predicted_class_name = CLASSES[predicted_class_index]
    return {
        'predicted_class_name': predicted_class_name,
        'confidence_score': confidence_score
    }
