import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dims = 8
# capacity = no. of hidden neurons in a layer
capacity = 8
variational_beta = 1

# Architecture of VAE Model
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        c = capacity
        self.fc1 = nn.Linear(in_features=input_dim, out_features=c * 2)
        self.fc2 = nn.Linear(in_features=c * 2, out_features=latent_dims)
        self.fc3 = nn.Linear(in_features=c * 2, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x_mu = self.fc2(x)
        x_logvar = self.fc3(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c * 2)
        self.fc_output = nn.Linear(in_features=c * 2, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x_recon = self.fc_output(x)
        return x_recon


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(output_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')  # Mean Squared Error loss for tabular data
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence