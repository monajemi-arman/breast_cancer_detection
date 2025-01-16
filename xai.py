#!/usr/bin/env python
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cloudpickle as pkl
from detectron2.modeling import build_model
from detectron import predict

detectron_cfg_pkl = 'detectron.cfg.pkl'

def main():
    argparser = ArgumentParser()
    argparser.add_argument('-i', '--image-path', type=str)
    argparser.add_argument('-w', '--weights-path', type=str)
    parsed = argparser.parse_args()

    with open(detectron_cfg_pkl, 'rb') as f:
        cfg = pkl.load(f)

    cfg.MODEL.WEIGHTS = parsed.weights_path

    predictions = predict(cfg, parsed, visualize=False)

    model = build_model(cfg)

    get_gradcam(model, predictions, parsed.image_path)

def get_gradcam(model, predictions, image_path, target_layer='backbone.bottom_up.res5.2.conv3'):
    """
    Generates and displays the Grad-CAM visualization for the top prediction.

    Args:
        model (torch.nn.Module): The pre-trained model.
        predictions (dict): The prediction results from Detectron.
        image_path (str): Path to the input image.
        target_layer (str, optional): The target layer for Grad-CAM. Defaults to 'backbone.bottom_up.res5.2.conv3'.
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Extract instances from predictions
    instances = predictions.get('instances', None)
    if instances is None or len(instances) == 0:
        print("No instances detected.")
        return

    # Get the top scoring instance
    top_instance = instances[0]
    class_id = top_instance.pred_classes.item()
    score = top_instance.scores.item()
    box = top_instance.pred_boxes.tensor.cpu().numpy()[0]
    print(f"Top prediction - Class ID: {class_id}, Score: {score:.4f}, Box: {box}")

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust based on your model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move model and input to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Initialize Grad-CAM
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # Generate Grad-CAM mask for the target class
    activation_map = cam_extractor(class_id, input_tensor)

    # Remove hooks to clean up
    cam_extractor.remove_hooks()

    # Post-process the input image for visualization
    image_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
    image_np = np.clip(image_np, 0, 1)
    image_pil = Image.fromarray((image_np * 255).astype('uint8'))

    # Overlay the Grad-CAM mask on the image
    mask = activation_map[0].cpu().numpy()
    result = overlay_mask(image_pil, mask, alpha=0.5)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.axis('off')
    plt.title(f"Grad-CAM for Class ID {class_id} with Score {score:.2f}")
    plt.show()



if __name__ == '__main__':
    main()