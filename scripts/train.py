import torch
import torch.nn as nn
import torch.optim as optimizer
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import Generator, Discriminator
from tqdm import tqdm


transform = transforms.Compose([
                                transforms.Resize((128, 128)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])

data_dir = "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/animals-20240921T054754Z-001/animals"
data_set = datasets.ImageFolder(root=data_dir, transform=transform)
batch_size = 64
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)


desired_device_index = 6
torch.cuda.set_device(desired_device_index)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator(100, 64).to(device)
disc = Discriminator(64).to(device) 

fixed_noise = torch.randn(32, 100, 1, 1).to(device)

learning_rate = 2e-4
optim_gen = optimizer.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_disc = optimizer.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
num_epochs = 100
criterian = nn.BCELoss()

gen.train()
disc.train()
step1 = 0
step2 = 0

writer_real = SummaryWriter(f"/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/tensorboard")
writer_fake = SummaryWriter(f"/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/tensorboard")
writer_loss_plot = SummaryWriter(f"/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/tensorboard")

gen_loss_vector = []
disc_loss_vector = []


for epoch in range(num_epochs):
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_idx, (real, _) in enumerate(data_loader):
            real = real.to(device)
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            fake = gen(noise)

            pred_real = disc(real).reshape(-1)
            loss_real = criterian(pred_real, torch.ones_like(pred_real))
            pred_fake = disc(fake.detach()).reshape(-1)
            loss_fake = criterian(pred_fake, torch.zeros_like(pred_fake))
            loss_disc = (loss_fake+loss_real)/2
            optim_disc.zero_grad()
            loss_disc.backward()
            disc_loss_vector.append(loss_disc.detach().cpu().numpy().item())
            optim_disc.step()

            writer_loss_plot.add_scalar("Discriminator loss", loss_disc, global_step=step2)

            output = disc(fake).reshape(-1)
            loss_gen = criterian(output, torch.ones_like(output))
            optim_gen.zero_grad()
            loss_gen.backward()
            gen_loss_vector.append(loss_gen.detach().cpu().numpy().item())
            optim_gen.step()
            writer_loss_plot.add_scalar(f"Generator loss", loss_gen, global_step=step2)

            step2 += 1
            pbar.update(1)

        
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step1)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step1)

                    step1 += 1

        
checkpoints_disc = {"Discriminator state dict": disc.state_dict()}
checkpoints_gen = {"Generator state dict": gen.state_dict()}


torch.save(checkpoints_disc, "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/checkpoints")
torch.save(checkpoints_gen, "/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/checkpoints")
np.save("/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/gen_loss.npy", gen_loss_vector)
np.save("/data6/shubham/PC/course_assignments/ADRL/Assignment-1/DC_GAN/results/disc_loss.npy", disc_loss_vector)






