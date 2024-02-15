'''
Get C2C_VAE based semi-factual
'''
import torch

# Function to find the nearest value in the dictionary for categorical features
def find_nearest_value(cat_embed, idx, value):
    values = cat_embed[idx].values()
    nearest_value = min(values, key=lambda x: abs(x - value))
    return nearest_value


def get_c2c_sf(query, label, cat_embed, device, vae, c2c_vae, c2c_latent_dims):

    N_guides = 4
    # A small 𝜆 value( < 0.5) means the output is more similar to the query
    lambd = 0.2

    # Convert numpy arrays to PyTorch tensors
    query_tensor = torch.tensor([query], dtype=torch.float32)

    label1 = label
    label2 = label

    label_diff = torch.cat((torch.tensor([[label1]]), torch.tensor([[label2]])), dim=1)
    label_diff = label_diff.to(device)

    with torch.no_grad():
        _, latent_mu1, _ = vae(query_tensor.to(device))

    latent_diffs = torch.empty(N_guides, c2c_latent_dims).normal_(mean=0, std=0.5).to(device)
    label_diff_repeated = label_diff.repeat(N_guides, 1)
    c2c_encoding = torch.cat((latent_diffs, label_diff_repeated.float()), dim=1)

    with torch.no_grad():
        c2c_recon = c2c_vae.decoder(c2c_encoding)
        recon_from_c2c_recon = vae.decoder(latent_mu1 - c2c_recon).cpu()

    queries = torch.cat([query_tensor] * N_guides, axis=0)
    # print(queries.shape)
    diff = (recon_from_c2c_recon - queries) ** 2
    # print(torch.sum(diff,axis=2).shape)
    guide_idx = torch.argmin(torch.sum(diff, dim=1))
    # print(recon_from_c2c_recon.shape)
    native_guide = recon_from_c2c_recon[guide_idx, :]
    # guide_latent = (latent_mu1 - c2c_recon)[guide_idx]

    with torch.no_grad():
        _, latent_mu2, _ = vae(native_guide.to(device).unsqueeze(0))
        new_latent_mu = (1 - lambd) * latent_mu1 + lambd * latent_mu2
        sf = vae.decoder(new_latent_mu)

    sf = sf[0].cpu().numpy()

    for idx in cat_embed:
        value = sf[idx]
        nearest_value = find_nearest_value(cat_embed, idx, value)
        sf[idx] = nearest_value

    return sf