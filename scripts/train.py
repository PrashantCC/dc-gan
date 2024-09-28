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

from model import Generator, Discriminator, Decoder



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Configuration
DATA_DIR = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/animals-20240921T054754Z-001/animals"
RESULTS_DIR = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results_decoder"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")

BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 200
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
decoder = Decoder().to(device)

#initialize parameters
initialize_weights(gen)
initialize_weights(disc)
initialize_weights(decoder)

# Optimizers
optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optim_decoder = optim.Adam(decoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Learning rate scheduler
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_decoder = optim.lr_scheduler.ExponentialLR(optim_decoder, gamma=0.99)

# Loss function
criterion = nn.BCELoss()
criterion_decoder = nn.MSELoss()

# Fixed noise for visualization
fixed_noise = torch.randn(32, Z_DIM).to(device)

# TensorBoard writers
writer_real = SummaryWriter(os.path.join(TENSORBOARD_DIR, "real"))
writer_fake = SummaryWriter(os.path.join(TENSORBOARD_DIR, "fake"))
writer_loss = SummaryWriter(os.path.join(TENSORBOARD_DIR, "loss"))

gen.train()
disc.train()
step = 0

start_epoch = 73
CHECKPOINT_PATH = f"/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results_decoder/checkpoints/checkpoint_epoch_{start_epoch}.pth"

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    gen.load_state_dict(checkpoint["generator"])
    disc.load_state_dict(checkpoint["discriminator"])
    decoder.load_state_dict(checkpoint["decoder"])
    optim_gen.load_state_dict(checkpoint["optim_gen"])
    optim_disc.load_state_dict(checkpoint["optim_disc"])
    optim_decoder.load_state_dict(checkpoint["opt_decoder"])

    # Set the start epoch for resuming training
    start_epoch = 0  # Start from the next epoch after 73
else:
    print("Checkpoint not found. Starting training from scratch.")
    start_epoch = 1

for epoch in range(74, NUM_EPOCHS+1):

    total_loss_gen = 0
    total_loss_dec = 0
    total_loss_gen_dec = 0
    total_loss_disc = 0
    batch_step = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, (real, _) in pbar:
        
        real = real.to(device)
        current_batch_size = real.size(0)
        
        # Train Discriminator
        noise = torch.randn(current_batch_size, Z_DIM).to(device)
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
        noise_pred = decoder(fake)
        loss_gen = criterion(output, torch.ones_like(output))
        loss_dec = criterion_decoder(noise, noise_pred)
        loss_gen_dec = loss_gen + 0.5 * loss_dec
        gen.zero_grad()
        decoder.zero_grad()
        loss_gen_dec.backward()

        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

        optim_gen.step()
        optim_decoder.step()
        
        total_loss_gen += loss_gen.item()
        total_loss_dec += loss_dec.item()
        total_loss_gen_dec += loss_gen_dec.item()
        total_loss_disc += loss_disc.item()
        
        # Update TensorBoard
        if batch_idx % 10 == 0:
            writer_loss.add_scalar("Discriminator Loss Batch", loss_disc.item(), global_step=batch_step)
            writer_loss.add_scalar("Generator Loss Batch", loss_gen.item(), global_step=batch_step)
            writer_loss.add_scalar("Decoder Loss Batch", loss_dec.item(), global_step=batch_step)
            writer_loss.add_scalar("Generator_Decoder Loss Batch", loss_gen_dec.item(), global_step=batch_step)
            batch_step += 1
        
        pbar.set_postfix({"Disc_Loss": loss_disc.item(), "Gen_dec_Loss": loss_gen_dec.item(), "G_Loss": loss_gen.item(), "dec_Loss": loss_dec.item()})
    
    with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        #get average loss of generator and critic for each epoch
        avg_loss_gen = total_loss_gen / len(dataloader)
        avg_loss_dec = total_loss_dec / len(dataloader)
        avg_loss_gen_dec = total_loss_gen_dec / len(dataloader)
        avg_loss_disc= total_loss_disc / len(dataloader)
        #write loss to tensorboard
        writer_loss.add_scalar("Generator loss Epoch", avg_loss_gen, global_step=step)
        writer_loss.add_scalar("Decoder loss Epoch", avg_loss_dec, global_step=step)
        writer_loss.add_scalar("Generator_Decoder loss Epoch", avg_loss_gen_dec, global_step=step)
        writer_loss.add_scalar("Discriminator loss Epoch", avg_loss_disc, global_step=step)
    

    step += 1

    # Update learning rates
    scheduler_g.step()
    scheduler_d.step()
    scheduler_decoder.step()
    
    # Save checkpoints
    torch.save({
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "decoder": decoder.state_dict(),
        "optim_gen": optim_gen.state_dict(),
        "optim_disc": optim_disc.state_dict(),
        "opt_decoder": optim_decoder.state_dict()
    }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

# Save final model and losses
torch.save(gen.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_final.pth"))
torch.save(disc.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_final.pth"))
torch.save(decoder.state_dict(), os.path.join(CHECKPOINT_DIR, "decoder_final.pth"))


writer_real.close()
writer_fake.close()
writer_loss.close()

print("Training completed.")  