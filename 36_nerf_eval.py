import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NerfModel, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        # density estimation
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma
    
def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('model.pth', map_location=device)
    testing_dataset = np.load('testing_data.pkl', allow_pickle=True)

    model = NerfModel(hidden_dim=256).to(device)
    model.load_state_dict(checkpoint)

    img_index = 20
    chunk_size = 10
    H = 400
    W = 400
    dataset = torch.from_numpy(testing_dataset).to(device)
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    for i in tqdm(range(int(np.ceil(H / chunk_size)))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        with torch.inference_mode():
            regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=2, hf=6, nb_bins=192)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    # print(np.min(img), np.max(img))
    img = np.clip(img, 0, 1)

    # Save the generated image
    plt.figure()
    plt.imshow(img)
    plt.title(f'Generated image [{img_index}]')
    plt.savefig(f'generated_image_{img_index}.png')
    plt.close()

    # Save the ground truth image
    ground_truth_px_values = dataset[img_index * H * W: (img_index + 1) * H * W, 6:9]
    img = ground_truth_px_values.reshape(H, W, 3).cpu().numpy()
    plt.figure()
    plt.imshow(img)
    plt.title(f'test image [{img_index}]')
    plt.savefig(f'test_image_{img_index}.png')
    plt.close()
   

    # Generate a video of the test dataset
    img_list = []
    for i in range(int(testing_dataset.shape[0] / (H * W))):
        batch = testing_dataset[i * H * W : (i + 1) * H * W]
        ground_truth_px_values = batch[:, 6:9]
        img = ground_truth_px_values.reshape(H, W, 3)
        img_list.append(img)
    
    fig, ax = plt.subplots()
    im = ax.imshow(img_list[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=img_list, interval=50, blit=True)
    plt.close()
    ani.save('test_dataset.mp4', fps=30)

    # Generate a video of the generated images
    img_list = []
    for img_index in tqdm(range(len(testing_dataset) // (H * W))):
        chunk_size = 10
        H = 400
        W = 400
        dataset = torch.from_numpy(testing_dataset).to(device)
        ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

        data = []   # list of regenerated pixel values
        for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
            # Get chunk of rays
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
            with torch.inference_mode():
                regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=2, hf=6, nb_bins=192)
            data.append(regenerated_px_values)
        img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
        img = np.clip(img, 0, 1)
        img_list.append(img)

    fig, ax = plt.subplots()
    im = ax.imshow(img_list[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=img_list, interval=50, blit=True)
    plt.close()
    ani.save('generated_images.mp4', fps=30)