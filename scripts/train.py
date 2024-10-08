import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm

from model import Generator, Discriminator



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Configuration
DATA_DIR = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/animals-20240921T054754Z-001/animals"
RESULTS_DIR = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")

BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 500
DESIRED_DEVICE_INDEX = 2
Z_DIM = 128
FEATURES_G = 64
FEATURES_D = 64
img_channel = 3

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
gen = Generator(img_channel, Z_DIM, FEATURES_G).to(device)
disc = Discriminator(img_channel, FEATURES_D).to(device)

#initialize parameters
initialize_weights(gen)
initialize_weights(disc)

# Optimizers
optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Learning rate scheduler
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)

# Loss function
criterion = nn.BCELoss()

# Fixed noise for visualization
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# TensorBoard writers
writer_real = SummaryWriter(os.path.join(TENSORBOARD_DIR, "real"))
writer_fake = SummaryWriter(os.path.join(TENSORBOARD_DIR, "fake"))
writer_loss = SummaryWriter(os.path.join(TENSORBOARD_DIR, "loss"))

gen.train()
disc.train()
step = 0

for epoch in range(NUM_EPOCHS):

    total_loss_gen = 0
    total_loss_disc = 0
    batch_step = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, (real, _) in pbar:
        
        real = real.to(device)
        current_batch_size = real.size(0)
        
        # Train Discriminator
        noise = torch.randn(current_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        optim_disc.step()
        
        # Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        optim_gen.step()
        
        total_loss_gen += loss_gen.item()
        total_loss_disc += loss_disc.item()
        
        # Update TensorBoard
        if batch_idx % 10 == 0:
            writer_loss.add_scalar("Discriminator Loss Batch", loss_disc.item(), global_step=batch_step)
            writer_loss.add_scalar("Generator Loss Batch", loss_gen.item(), global_step=batch_step)
            batch_step += 1
        
        pbar.set_postfix({"D Loss": loss_disc.item(), "G Loss": loss_gen.item()})
    
    with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        #get average loss of generator and critic for each epoch
        avg_loss_gen = total_loss_gen / len(dataloader)
        avg_loss_disc= total_loss_disc / len(dataloader)
        #write loss to tensorboard
        writer_loss.add_scalar("Generator loss Epoch", avg_loss_gen, global_step=step)
        writer_loss.add_scalar("Discriminator loss Epoch", avg_loss_disc, global_step=step)

    step += 1

    # Update learning rates
    scheduler_g.step()
    scheduler_d.step()
    
    # Save checkpoints
    torch.save({
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "optim_gen": optim_gen.state_dict(),
        "optim_disc": optim_disc.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

# Save final model and losses
torch.save(gen.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_final.pth"))
torch.save(disc.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_final.pth"))


writer_real.close()
writer_fake.close()
writer_loss.close()

print("Training completed.")  