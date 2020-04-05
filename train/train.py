import argparse
import json
import os
import pickle
import sys
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pickle as pkl
from torchvision import transforms
from torchvision import datasets


from model import DataDiscriminator, DataGenerator



def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #D = DataDiscriminator(64)
    G = DataGenerator(z_size=100, conv_dim=64)
    
    model_info = {}
    model_info_path = os.path.join(model_dir, 'generator_model.pt')
    
    with open(model_info_path, 'rb') as f:
        G.load_state_dict(torch.load(f))

    G.to(device).eval()

    print("Done loading model.")
    return G



def real_loss(D_out,train_on_gpu, smooth=False):
    
    batch_size = D_out.size(0)
    
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    
    min, max = feature_range
    x = x * (max - min) + min
    
    return x


def get_dataloader(batch_size, image_size, data_dir):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    # TODO: Implement function and return a dataloader
    
    transform = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor()])
    
    dataset = datasets.ImageFolder(data_dir,transform=transform)
    
    #rand_sampler = torch.utils.data.RandomSampler(dataset, num_samples=32, replacement=True)
    #dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size,shuffle=False, sampler=rand_sampler)
    
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size,shuffle=True)
        
    return dataloader


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    if classname =='Conv2d':
        
        w = torch.empty(m.weight.data.shape)
        m.weight.data = nn.init.kaiming_uniform_(w)
        
    if classname =='Linear':
        w = torch.empty(m.weight.data.shape)
        m.weight.data = nn.init.kaiming_uniform_(w)     


def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = DataDiscriminator(d_conv_dim)
    G = DataGenerator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G


def real_loss(D_out,train_on_gpu, smooth=False):
    
    batch_size = D_out.size(0)
    
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    
    #print(D_out.squeeze().shape)
    #print(labels.shape)
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def train(D, G, z_size, train_loader, epochs, d_optimizer, g_optimizer, train_on_gpu):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    
    print_every=50
    losses = []
    
    if train_on_gpu:
        D.cuda()
        G.cuda()
    
    samples = []
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    for epoch in range(epochs):

        for batch_i, (real_images,_) in enumerate(train_loader):
            
            batch_size = real_images.size()[0]

            real_images = scale(real_images)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            
            D.train()
            G.train()
            
            d_optimizer.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images 
            if train_on_gpu:
                real_images = real_images.cuda()
                
            D_real = D(real_images)

            d_real_loss = real_loss(D_real, train_on_gpu)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_data = G(z)

            # Compute the discriminator losses on fake images            
            D_fake = D(fake_data)
            d_fake_loss = fake_loss(D_fake, train_on_gpu)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()


            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, train_on_gpu) # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        # generate and save sample, fake images
        G.eval() # for generating samples
        if train_on_gpu:
            fixed_z = fixed_z.cuda()
            
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode
    
    # _ = view_samples(-1, samples_z)
    
    print(samples_z)
    
    with open('generator_model.pt', 'wb') as f:
        torch.save(G.state_dict(),f )
    
    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
        
        


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--z_size', type=int, default=100, metavar='N',
                        help='input z-size for training (default: 100)')
    
    # Model Parameters
    parser.add_argument('--conv_dim', type=int, default=64, metavar='N',
                        help='size of the convolution dim (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='N',
                        help='Learning rate default 0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='N',
                        help='beta1 default value 0.5')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='N',
                        help='beta2 default value 0.999')
    parser.add_argument('--img_size', type=int, default=32, metavar='N',
                        help='Image size default value 32')
    
    
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus',type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.cuda.is_available()
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = get_dataloader(args.batch_size, args.img_size, args.data_dir)
    
    
    # Build the model.
    
    D, G = build_network(args.conv_dim, args.conv_dim, z_size=args.z_size)
    
    #D = DataDiscriminator(args.conv_dim)
    #G = DataGenerator(z_size=args.z_size, conv_dim=args.conv_dim)
    

    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(D.parameters(), args.lr, [args.beta1, args.beta2])
    g_optimizer = optim.Adam(G.parameters(), args.lr, [args.beta1, args.beta2])
    
    train(D, G, args.z_size, train_loader, args.epochs, d_optimizer, g_optimizer, device)

	# Save the model parameters
    G_path = os.path.join(args.model_dir, 'generator_model_main.pt')
    with open(G_path, 'wb') as f:
        torch.save(G.cpu().state_dict(), f)


