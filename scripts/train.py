import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
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

# Split the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize models
decoder = Decoder().to(device)
classifier = Classifier().to(device)

# Optimizers
optim_classi = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Learning rate scheduler
scheduler_classi = optim.lr_scheduler.ExponentialLR(optim_classi, gamma=0.99)

# Loss function
criterion = nn.CrossEntropyLoss()

# TensorBoard writer
writer = SummaryWriter(TENSORBOARD_DIR)

# Load pre-trained decoder

decoder_checkpoint_path = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results_decoder/checkpoints/decoder_final.pth"
decoder.load_state_dict(torch.load(decoder_checkpoint_path))
decoder.eval()

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            decoded_inputs = decoder(inputs)
        outputs = model(decoded_inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'accuracy': f"{correct_preds/total_preds:.4f}"})
    
    return running_loss / len(loader), correct_preds / total_preds

# Testing function
def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            decoded_inputs = decoder(inputs)
            outputs = model(decoded_inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    
    return running_loss / len(loader), correct_preds / total_preds

# Main training loop
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    train_loss, train_acc = train_epoch(classifier, train_loader, optim_classi, criterion, device)
    test_loss, test_acc = test(classifier, test_loader, criterion, device)
    
    # Logging to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Test', test_acc, epoch)
    
    # Log learning rate
    writer.add_scalar('Learning Rate', scheduler_classi.get_last_lr()[0], epoch)
    
    # Print progress
    tqdm.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Step the scheduler
    scheduler_classi.step()

# Close the writer
writer.close()

# Save the final model
torch.save(classifier.state_dict(), os.path.join(CHECKPOINT_DIR, "classifier_final.pth"))
print("Training completed. Model saved.")
print(f"TensorBoard logs saved to {TENSORBOARD_DIR}")
print("To view the TensorBoard, run:")
print(f"tensorboard --logdir={TENSORBOARD_DIR}")