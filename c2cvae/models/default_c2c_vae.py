import torch
import torch.nn as nn
import torch.nn.functional as F

label_size = 2  # Assuming two classes

#from vae
latent_dims = 10

c2c_latent_dims = latent_dims - 1
c2c_capacity = latent_dims
c2c_variational_beta = 1

# Architecture of C2C_VAE Model

class c2c_Encoder(nn.Module):
    def __init__(self):
        super(c2c_Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=latent_dims + label_size, out_features=c2c_capacity)
        self.fc_mu = nn.Linear(in_features=c2c_capacity, out_features=c2c_latent_dims)
        self.fc_logvar = nn.Linear(in_features=c2c_capacity, out_features=c2c_latent_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class c2c_Decoder(nn.Module):
    def __init__(self):
        super(c2c_Decoder, self).__init__()
        self.fc2 = nn.Linear(in_features=c2c_latent_dims + label_size, out_features=c2c_capacity)
        self.fc1 = nn.Linear(in_features=c2c_capacity, out_features=latent_dims)

    def forward(self, x):
        x = self.fc2(x)
        x = self.fc1(x)
        return x


class c2c_VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(c2c_VariationalAutoencoder, self).__init__()
        self.encoder = c2c_Encoder()
        self.decoder = c2c_Decoder()

    def forward(self, x):
        x, label = torch.split(x, (latent_dims, label_size), dim=1)
        c2c_latent_mu, c2c_latent_logvar = self.encoder(torch.cat((x, label), dim=1))
        c2c_latent = self.latent_sample(c2c_latent_mu, c2c_latent_logvar)
        x_recon = self.decoder(torch.cat((c2c_latent, label), dim=1))
        return x_recon, c2c_latent_mu, c2c_latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def c2c_vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x.view(-1, latent_dims), x.view(-1, latent_dims), reduction='sum')

    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + c2c_variational_beta * kldivergence