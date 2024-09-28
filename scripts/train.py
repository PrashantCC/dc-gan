import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from model import Decoder, Classifier  # Assuming these are defined in model.py

# Configuration
DATA_DIR = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/animals-20240921T054754Z-001/animals"
RESULTS_DIR = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results_decoder_classifier"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")

BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 200
DESIRED_DEVICE_INDEX = 0

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Set up device
torch.cuda.set_device(DESIRED_DEVICE_INDEX)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Initialize models
decoder = Decoder().to(device)
classifier = Classifier().to(device)

# Optimizers
optim_classi = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Learning rate scheduler
scheduler_classi = optim.lr_scheduler.ExponentialLR(optim_classi, gamma=0.99)

# Loss function
criterion = nn.CrossEntropyLoss()

# TensorBoard writers
writer_loss_accuracy = SummaryWriter(os.path.join(TENSORBOARD_DIR, "loss_accuracy"))

# Load pre-trained decoder
decoder_checkpoint_path = os.path.join(CHECKPOINT_DIR, "decoder_final.pth")
decoder.load_state_dict(torch.load(decoder_checkpoint_path))
decoder.eval()

classifier.train()

# Main training loop with progress bar
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Inner loop with progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optim_classi.zero_grad()

        # Forward pass
        with torch.no_grad():
            decoded_inputs = decoder(inputs)
        outputs = classifier(decoded_inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optim_classi.step()

        # Update loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'accuracy': f"{correct_preds/total_preds:.4f}"
        })

    # Average loss and accuracy for the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / total_preds

    # Logging to TensorBoard
    writer_loss_accuracy.add_scalar('Loss/train', epoch_loss, epoch)
    writer_loss_accuracy.add_scalar('Accuracy/train', epoch_accuracy, epoch)

    # Print progress
    tqdm.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Step the scheduler
    scheduler_classi.step()

# Close the writer
writer_loss_accuracy.close()

# Save the final model
torch.save(classifier.state_dict(), os.path.join(CHECKPOINT_DIR, "classifier_final.pth"))
print("Training completed. Model saved.")