import argparse
import json
import os
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

num_classes = 2

torch.set_float32_matmul_precision('medium')

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class JSONImageDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=default_transform):
        self.img_dir = Path(img_dir)
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.data[idx]['file_name']
        label = self.data[idx]['class_id']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}")

def predict_image(model, img_path):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    transformed_image = default_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted_class = torch.max(output, 1)
    print(f"Predicted Class: {predicted_class.item()}")

    cam_extractor = SmoothGradCAMpp(model.model)
    activation_map = cam_extractor(predicted_class.item(), output)[0]
    result = overlay_mask(to_pil_image(transformed_image[0]), to_pil_image(activation_map, mode='F'), alpha=0.5)

    plt.imshow(result)
    plt.axis('off')
    plt.show()

def main():
    global num_classes

    parser = argparse.ArgumentParser(description="Train, evaluate, or predict using an image classifier.")
    parser.add_argument("-c", "--command", type=str, required=True, choices=["train", "evaluate", "predict"], help="Command to execute.")
    parser.add_argument("-a", "--annotations", type=str, help="Path to JSON file containing images and class IDs.")
    parser.add_argument("-d", "--img_dir", type=str, help="Directory containing images.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save TensorBoard data and checkpoints.")
    parser.add_argument("-i", "--input_image", type=str, help="Path to input image for prediction.")

    args = parser.parse_args()

    if args.command == "train":
        dataset = JSONImageDataset(args.annotations, args.img_dir)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        num_classes = len(set(item['class_id'] for item in dataset.data))

        model = ImageClassifier(num_classes)

        early_stopping = EarlyStopping(monitor='train_loss', patience=args.patience, mode='min')
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=-1,
            every_n_epochs=5
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            default_root_dir=args.save_dir
        )

        trainer.fit(model, dataloader)

    elif args.command == "evaluate":
        dataset = JSONImageDataset(args.annotations, args.img_dir)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        model = ImageClassifier(num_classes)
        model.load_from_checkpoint(Path(args.save_dir) / "last.ckpt")

        evaluate_model(model, dataloader)

    elif args.command == "predict":
        model = ImageClassifier.load_from_checkpoint(Path(args.save_dir) / "last.ckpt").to('cpu')

        predict_image(model, args.input_image)

if __name__ == "__main__":
    main()

