import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from torchvision.transforms.functional import pad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.colors as mcolors




def load_model(model_path):
    # Initialize the architecture
    model = smp.Unet('mobilenet_v2', encoder_weights=None, classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)
    

def predict_image_mask(model, image):
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        masked = torch.argmax(output, dim=1).squeeze(0)
    return masked

def predict_image_mask_miou(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)

    # Ensure the image dimensions are divisible by 32
    h, w = image.shape[1:]
    pad_h = (32 - h%32) if h%32 != 0 else 0
    pad_w = (32 - w%32) if w%32 != 0 else 0
    image = pad(image, padding=(0, 0, pad_w, pad_h))  # padding format is (left, top, right, bottom)

    model.to(device); image=image.to(device)
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked


def display_images(original, mask):
    # Check if mask is a tensor
    if torch.is_tensor(mask):
        # Move the mask tensor to cpu and convert to numpy array
        mask = mask.cpu().numpy()

        # If mask has multiple channels
        if mask.ndim == 3:
            # Transpose from (C,H,W) to (H,W,C)
            mask = np.transpose(mask, (1, 2, 0))

        # Ensure the mask is contiguous
        mask = np.ascontiguousarray(mask)

    # Apply colormap
    cmap = plt.get_cmap('viridis')
    mask_rgb = cmap(mask)[:, :, :3]  # Keep only RGB channels

    # Convert mask back to PIL Image
    mask_pil = Image.fromarray((mask_rgb * 255).astype(np.uint8))

    # Display the images
    st.image(original, caption='Original Image', use_column_width=True)
    st.image(mask_pil, caption='Predicted Mask', use_column_width=True)




# Main app
def main():
    # Load model
    model = load_model('Unet-Mobilenet.pt')

    # Title
    st.title("Drone Image Segmentation")

    # Upload the image
    image_file = st.file_uploader("Upload Drone Image", type=['jpg', 'png'])

    if image_file is not None:
        # Convert the file to an OpenCV image
        image = Image.open(image_file)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform prediction
        mask = predict_image_mask(model, preprocessed_image)

        # Display original and predicted images
        display_images(opencv_image, mask)

if __name__ == "__main__":
    main()