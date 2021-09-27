!pip install av

# imports etc

!pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

import os
import sys
import time
import json
import glob
import torch
from torch import nn
from torch.nn import functional as F
import math
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    VolumeRenderer,
    NDCGridRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.transforms import so3_exponential_map

from torch.utils.data import Dataset
import os
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import random
from torchvision.io import read_video
from torchvision.io import write_video
from torchvision.io import write_jpeg


# add path for demo utils functions 
sys.path.append(os.path.abspath(''))
!wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/docs/tutorials/utils/plot_image_grid.py
from plot_image_grid import image_grid

# obtain the utilized device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(device)

!unzip Recordings_Thicker.zip

from torchvision.io import read_video

datasetfolder = 'Recordings_Thicker'
batchsize = 1
numviews = 1
volume_width = 1
z_size = 128
generator_latent_n_features_min = 64
discriminator_n_features_min = 64
voxels_side = 64
# PE_dim = 3 + 6*10 # number of positional encoding channels
# PE_dim_first = 3 + 6*10
PE_dim = 0 # number of positional encoding channels
PE_dim_first = 0
const_w_size = 128
image_size=64
mindepth=0.4
maxdepth=1.6
num_frames = 16
GP_lambda = 10

random.seed(0)
torch.manual_seed(0)

class VideoDataset(Dataset):
    def __init__(self,foldername,extension):
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(foldername):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        self.file_names = list()

        for elem in listOfFiles:
          if elem.endswith(extension):
            self.file_names += [elem]

        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length
        # return 1

    def __getitem__(self, idx):
        video_path = self.file_names[idx]
        video, audio, info = read_video(video_path, pts_unit='pts')
        return video.permute(3,1,2,0) / 255


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4) or (len(size) == 5)
    N, C = size[0:2]
    feat_var = feat.view(N, C, -1).var(unbiased=False, dim=2) + eps
    view_size = (N, C,) + (1,) * (len(size) - 2)
    feat_std = feat_var.sqrt().view(*view_size)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(*view_size)
    return feat_mean, feat_std

def AdaIn(features, scale, bias,k=False):
    size = features.size()
    content_mean, content_std = calc_mean_std(features)
    normalized_feat = (features - content_mean.expand(size)) / content_std.expand(size)
    N, C = size[0:2]
    view_size = (N, C,) + (1,) * (len(size) - 2)
    scale = scale.view(*view_size)
    bias = bias.view(*view_size)
    normalized = normalized_feat * scale.expand(size) + bias.expand(size)
    return normalized

def AddPositionalEncoding(input,first=True):
    return input
    # batch_size, _, dim, _,_ = input.size()
    # p = (torch.arange(dim).float() / (dim - 1))  * 2 - 1
    # x = p.repeat(1,dim,dim,1)
    # x = x.repeat(batch_size,1,1,1,1).to(device)
    # y = x.transpose(2,4)
    # z = x.transpose(3,4)
    # output = torch.cat([input,x,y,z], dim=1)
    # for i in range(10):
    #   sinx = torch.sin(2**i * np.pi * x)
    #   cosx = torch.cos(2**i * np.pi * x)
    #   siny = torch.sin(2**i * np.pi * y)
    #   cosy = torch.cos(2**i * np.pi * y)
    #   sinz = torch.sin(2**i * np.pi * z)
    #   cosz = torch.cos(2**i * np.pi * z)
    #   output = torch.cat([output,sinx,cosx,siny,cosy,sinz,cosz],dim=1)
    # return output

class Generator_Latent(nn.Module):
    def __init__(self):
        super(Generator_Latent, self).__init__()
        const_w_std = 1

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.const_w = nn.Parameter(torch.randn(1, const_w_size,2,2,2) * const_w_std)
        self.conv1 = nn.Conv3d(const_w_size, generator_latent_n_features_min * 16, 3, 1, 1, bias=False)
        self.zmap1 = nn.Linear(z_size,generator_latent_n_features_min * 16 * 2)
        self.conv2 = nn.Conv3d(generator_latent_n_features_min * 16 + PE_dim, generator_latent_n_features_min * 8, 3, 1, 1, bias=False)
        self.zmap2 = nn.Linear(z_size,generator_latent_n_features_min * 8 * 2)
        self.conv3 = nn.Conv3d(generator_latent_n_features_min * 8 + PE_dim, generator_latent_n_features_min * 4, 3, 1, 1, bias=False)
        self.zmap3 = nn.Linear(z_size,generator_latent_n_features_min * 4 * 2)
        self.conv4 = nn.Conv3d(generator_latent_n_features_min * 4 + PE_dim, generator_latent_n_features_min * 2, 3, 1, 1, bias=False)
        self.zmap4 = nn.Linear(z_size,generator_latent_n_features_min * 2 * 2)

    def forward(self, z):
        x = self.const_w.expand(batchsize, const_w_size,2,2,2)
        s, b = torch.split(F.softplus(self.zmap1(z)), [generator_latent_n_features_min * 16, generator_latent_n_features_min * 16], dim=-1)
        x = F.softplus(AdaIn(self.conv1(x), s, b))
        x = self.upsample(x)
        s, b = torch.split(F.softplus(self.zmap2(z)), [generator_latent_n_features_min * 8, generator_latent_n_features_min * 8], dim=-1)
        x = F.softplus(AdaIn(self.conv2(AddPositionalEncoding(x,first=True)), s, b))
        x = self.upsample(x)
        s, b = torch.split(F.softplus(self.zmap3(z)), [generator_latent_n_features_min * 4, generator_latent_n_features_min * 4], dim=-1)
        x = F.softplus(AdaIn(self.conv3(AddPositionalEncoding(x)), s, b))
        x = self.upsample(x)
        s, b = torch.split(F.softplus(self.zmap4(z)), [generator_latent_n_features_min * 2, generator_latent_n_features_min * 2], dim=-1)
        x = F.softplus(AdaIn(self.conv4(AddPositionalEncoding(x)), s, b))
        return x

class Generator_Explicit(nn.Module):
    def __init__(self):
        super(Generator_Explicit, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        out_channels = 4 # density, rgb
        self.conv1 = nn.Conv3d(generator_latent_n_features_min * 2 + PE_dim, generator_latent_n_features_min * 1, 3, 1, 1, bias=False)
        self.zmap1 = nn.Linear(z_size,generator_latent_n_features_min * 1 * 2)
        self.conv2 = nn.Conv3d(generator_latent_n_features_min * 1 + PE_dim, out_channels, 3, 1, 1)

    def forward(self, z, state):
        x = self.upsample(state)
        s, b = torch.split(F.softplus(self.zmap1(z)), [generator_latent_n_features_min * 1, generator_latent_n_features_min * 1], dim=-1)
        x = F.softplus(AdaIn(self.conv1(AddPositionalEncoding(x)), s, b,k=True))
        x = self.upsample(x)
        x = self.conv2(AddPositionalEncoding(x))
        x -= 4
        x = torch.sigmoid(x)
        return x

class Change_World_State(nn.Module):
    def __init__(self):
        super(Change_World_State, self).__init__()
        self.conv1 = nn.Conv3d(generator_latent_n_features_min * 2 + PE_dim, generator_latent_n_features_min * 2, 3, 1, 1, bias=True)
        self.zmap1 = nn.Linear(z_size,generator_latent_n_features_min * 2 * 2)
        self.conv2 = nn.Conv3d(generator_latent_n_features_min * 2 + PE_dim, generator_latent_n_features_min * 2, 3, 1, 1, bias=True)
        self.zmap2 = nn.Linear(z_size,generator_latent_n_features_min * 2 * 2)
        self.conv3 = nn.Conv3d(generator_latent_n_features_min * 2 + PE_dim, generator_latent_n_features_min * 2, 3, 1, 1, bias=True)
        self.zmap3 = nn.Linear(z_size,generator_latent_n_features_min * 2 * 2)
        self.conv4 = nn.Conv3d(generator_latent_n_features_min * 2 + PE_dim, generator_latent_n_features_min * 2, 3, 1, 1, bias=True)
        self.zmap4 = nn.Linear(z_size,generator_latent_n_features_min * 2 * 2)

    def forward(self, z, state):
        s, b = torch.split(F.softplus(self.zmap1(z)), [generator_latent_n_features_min * 2, generator_latent_n_features_min * 2], dim=-1)
        x = F.softplus(AdaIn(self.conv1(AddPositionalEncoding(state)), s, b,k=True))
        s, b = torch.split(F.softplus(self.zmap2(z)), [generator_latent_n_features_min * 2, generator_latent_n_features_min * 2], dim=-1)
        x = F.softplus(AdaIn(self.conv2(AddPositionalEncoding(x)), s, b,k=True))
        s, b = torch.split(F.softplus(self.zmap3(z)), [generator_latent_n_features_min * 2, generator_latent_n_features_min * 2], dim=-1)
        x = F.softplus(AdaIn(self.conv3(AddPositionalEncoding(x)), s, b,k=True))
        s, b = torch.split(F.softplus(self.zmap4(z)), [generator_latent_n_features_min * 2, generator_latent_n_features_min * 2], dim=-1)
        x = F.softplus(AdaIn(self.conv4(AddPositionalEncoding(x)), s, b,k=True))
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1, self.bn1 = self.get_conv(3, discriminator_n_features_min, 4, 2, 1)
        self.conv2, self.bn2 = self.get_conv(discriminator_n_features_min, discriminator_n_features_min * 2, 4, 2, 1)
        self.conv3, self.bn3 = self.get_conv(discriminator_n_features_min * 2, discriminator_n_features_min * 4, 4, 2, 1)
        self.conv4, self.bn4 = self.get_conv(discriminator_n_features_min * 4, discriminator_n_features_min * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(discriminator_n_features_min * 8,1,4, 1, 0)
        
        # self.conv5 = nn.Conv2d(discriminator_n_features_min * 8,discriminator_n_features_min * 8,4, 2, 1)
        # self.bn5 = nn.InstanceNorm2d(discriminator_n_features_min * 8,affine=True)
        # self.conv5 = nn.Conv2d(discriminator_n_features_min * 8, 1, 4, 1, 0)

        # self.conv4, self.bn4 = self.get_conv(discriminator_n_features_min * 4, discriminator_n_features_min * 8, 4, 2, 1)
        # self.conv5 = nn.Conv3d(discriminator_n_features_min * 8, 1, 4, 1, 0)

    def get_conv(self, n_features_in, n_features_out, kernel, stride, padding):
        return nn.Conv3d(n_features_in, n_features_out, kernel, stride, padding), nn.InstanceNorm3d(n_features_out,affine=True)

    def forward(self, input):
        # 128/64
        x = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.squeeze(x,dim=-1)
        x = self.conv5(x)


        # print(x.shape)
        # x_f2 = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        # print(x.shape)
        # x = self.conv5(x_f2)
        # print(x.shape)
        return x

def gradient_penalty(critic, real, fake):
  batch_size,C,H,W,D = real.shape
  epsilon = torch.rand((batch_size,1,1,1,1)).repeat(1,C,H,W,D).to(device)
  interpolated = real * epsilon + fake * (1-epsilon)
  mixed_scores = critic(interpolated)
  grads = torch.autograd.grad(
      inputs=interpolated, 
      outputs=mixed_scores,
      grad_outputs=torch.ones_like(mixed_scores),
      create_graph=True,
      retain_graph=True
  )[0]
  grads = grads.view(grads.shape[0],-1)
  grad_norm = grads.norm(2,dim=1)
  grad_penalty = torch.mean((grad_norm-1)**2)
  return grad_penalty

def adv_loss(prediction, target):
  targets = prediction.new_full(size=prediction.size(), fill_value=target)
  lossfunc = torch.nn.BCEWithLogitsLoss()
  loss = lossfunc(prediction, targets)
  return loss

dataset = VideoDataset(datasetfolder,'.mp4')
loader = DataLoader(dataset,batch_size=batchsize*numviews,shuffle=True)

# setup renderer
raysampler = NDCGridRaysampler(
    image_width=image_size,
    image_height=image_size,
    n_pts_per_ray=400,
    min_depth=mindepth,
    max_depth=maxdepth
)
raymarcher = EmissionAbsorptionRaymarcher()
renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher, sample_mode='bilinear')

# train
generator_latent = Generator_Latent().to(device)
generator_explicit = Generator_Explicit().to(device)
change_world_state = Change_World_State().to(device)
critic = Critic().to(device)

goptimiser = torch.optim.Adam(list(generator_latent.parameters())+list(generator_explicit.parameters())+list(change_world_state.parameters()), lr=0.0001, betas=(0.0, 0.9))
doptimiser = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))
gen_update_counter = 1
generator_loss = -1

def render_volume_batch(vol,cameras):
    densities = vol[:,0:1,:,:,:]
    colors = vol[:,1:,:,:,:]
    volumes = Volumes(
      densities = densities,
      features = colors,
      voxel_size = torch.ones(batchsize,1) * (volume_width / voxels_side),
    )
    rendered_images, rendered_silhouettes = renderer(cameras=cameras, volumes=volumes)[0].split([3, 1], dim=-1)
    rendered_images = rendered_images.permute(0,3,1,2)
    return rendered_images

for epoch in range(60):
  for idx, real in enumerate(loader):
    # setup novel cameras
    Rs = torch.zeros(numviews*batchsize,3,3)
    Ts = torch.zeros(numviews*batchsize,3)
    for i in range(numviews*batchsize):
      range_decider = random.uniform(0,1)
      if range_decider < 0.75:
        R, T = look_at_view_transform(dist=1, elev=random.uniform(-20,20), azim=random.uniform(0,360))
      else:
        R, T = look_at_view_transform(dist=1, elev=random.uniform(20,89), azim=random.uniform(0,360))
      # R, T = look_at_view_transform(dist=1, elev=0, azim=0)
      Rs[i,:,:] = R
      Ts[i,:] = T
    novelcameras = FoVPerspectiveCameras(R = Rs, T = Ts, znear = mindepth,zfar = maxdepth, aspect_ratio = 1.0,fov = 60.0,device = device)

    # generate video
    z = torch.randn(batchsize,z_size,device=device)
    state = generator_latent(z)
    explicit = generator_explicit(z,state)
    render_list = []
    render_list_spin = []
    render_list_mp4 = []
    mp4frame = 0

    for i in range(num_frames):
      if i>0:
        state = change_world_state(z,state)
        explicit = generator_explicit(z,state)
      renders = render_volume_batch(explicit, novelcameras)
      render_list.append(renders)
      # setup spin cameras for this frame
      spinRs = torch.zeros(numviews*batchsize,3,3)
      spinTs = torch.zeros(numviews*batchsize,3)
      for j in range(numviews*batchsize):
        R, T = look_at_view_transform(dist=1, elev=10, azim=i*36)
        spinRs[j,:,:] = R
        spinTs[j,:] = T
      spincameras = FoVPerspectiveCameras(R = spinRs, T = spinTs, znear = mindepth,zfar = maxdepth,aspect_ratio = 1.0,fov = 60.0,device = device)
      renders = render_volume_batch(explicit.detach(), spincameras)
      render_list_spin.append(renders)
      # render mp4 frames
      if idx % 200 == 0:
        for mi in range(8):
          mp4Rs = torch.zeros(batchsize,3,3)
          mp4Ts = torch.zeros(batchsize,3)
          for j in range(batchsize):
            R, T = look_at_view_transform(dist=1, elev=20 - np.sin(mp4frame/30)*20, azim=mp4frame * 3)
            mp4Rs[j,:,:] = R
            mp4Ts[j,:] = T
          mp4cameras = FoVPerspectiveCameras(R = mp4Rs, T = mp4Ts, znear = mindepth,zfar = maxdepth,aspect_ratio = 1.0,fov = 60.0,device = device)
          renders = render_volume_batch(explicit.detach(), mp4cameras)
          render_list_mp4.append(renders)
          mp4frame+=1

    space_time_block = torch.stack(render_list,dim=4)
    space_time_block_spin = torch.stack(render_list_spin,dim=4)
    if idx % 200 == 0:
      space_time_block_mp4 = torch.stack(render_list_mp4,dim=4)

    # critic training
    critic_real = critic(real.to(device)).reshape(-1)
    critic_fake = critic(space_time_block).reshape(-1)
    gp = gradient_penalty(critic, real.to(device), space_time_block)
    discriminator_loss = (torch.mean(critic_fake) - torch.mean(critic_real)) + GP_lambda * gp
    doptimiser.zero_grad()
    discriminator_loss.backward(retain_graph=True)
    doptimiser.step()

    # generator training
    if gen_update_counter == 5:
      gen_update_counter = 0
      critic_fake = critic(space_time_block).reshape(-1)
      generator_loss = -torch.mean(critic_fake)
      goptimiser.zero_grad()
      generator_loss.backward()
      goptimiser.step()
    gen_update_counter += 1

    # display intermediate results
    if idx % 200 == 0:
      # write video at higher resolution
      vid = (space_time_block_mp4[0,:,:,:,:]*255).cpu().type(torch.uint8).permute(3,0,1,2)
      vid = F.interpolate(vid,512)
      vid = vid.permute(0,2,3,1)
      write_video('wgan-gp_{}_{}.mp4'.format(epoch,idx),vid.cpu(),30,'libx264')
      # write video at real resolution
      # space_time_block_mp4 = space_time_block_mp4[0,:,:,:,:].permute(3,1,2,0)
      # space_time_block_mp4 = (space_time_block_mp4 * 255).type(torch.uint8)
      # write_video('wgan-gp_{}_{}.mp4'.format(epoch,idx),space_time_block_mp4.cpu(),30,'libx264')

      clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
      f, axarr = plt.subplots(3,16, figsize=(22, 6))
      [axi.set_axis_off() for axi in axarr.ravel()]
      for frame in range(num_frames):
        axarr[0,frame].imshow(clamp_and_detach(real[0,:,:,:,frame].permute(1,2,0)))
        axarr[1,frame].imshow(clamp_and_detach(space_time_block[0,:,:,:,frame].permute(1,2,0)))
        axarr[2,frame].imshow(clamp_and_detach(space_time_block_spin[0,:,:,:,frame].permute(1,2,0)))

      plt.show()
      print('epoch: ',epoch,'idx: ',idx)
      print('gen loss: ',generator_loss)
      print('dis loss: ',discriminator_loss)

