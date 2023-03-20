from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn.functional import relu
import numpy

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_dim, resolution):
        super().__init__()
        
        self.fc   = nn.Linear(4, hidden_dim, bias=True)
        self.grid = self.build_grid(resolution)
        
    def build_grid(self, resolution):
        ranges = [numpy.linspace(0, 1, res) for res in resolution]
        grid = numpy.meshgrid(*ranges, sparse=False, indexing='ij')
        grid = numpy.stack(grid, axis=-1)
        grid = grid.reshape(resolution[0], resolution[1], -1)
        grid = numpy.expand_dims(grid, axis=0)
        grid = grid.astype('float32')
        
        return torch.from_numpy(numpy.concatenate([grid, 1 - grid], axis=-1))
    
    def forward(self, x):
        return x + self.fc(self.grid)

class Encoder(nn.Module):
    def __init__(self, resolution, hidden_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(21, hidden_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2)
        
        self.soft_position_embed = SoftPositionEmbed(hidden_dim, resolution)
        
    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        x = self.conv4(x)
        x = relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.soft_position_embed(x)
        x = torch.flatten(x, 1, 2)
        
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim, resolution):
        super().__init__()
        self.resolution = resolution
        
        self.conv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hidden_dim, 4, 3, stride=(1, 1), padding=1)
        
        self.soft_position_embed = SoftPositionEmbed(hidden_dim, (8, 8))
        
    def forward(self, x):
        x = self.soft_position_embed(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        x = self.conv4(x)
        x = relu(x)
        x = self.conv5(x)
        x = relu(x)
        x = self.conv6(x)
        x = x[:, :, :self.resolution[0], :self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        
        return x
    
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_iterations, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots      = num_slots
        self.num_iterations = num_iterations
        
        self.scale = dim ** -0.5
        self.eps   = eps
        
        self.slots_mu    = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRUCell(dim, dim)
        
        hidden_dim = max(dim, hidden_dim)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, n, d = x.shape
        
        mu    = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_sigma.expand(b, self.num_slots,  -1)
        slots = torch.normal(mu, sigma)
        
        x = self.norm_input(x)
        k, v = self.to_k(x), self.to_v(x)
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attention = dots.softmax(dim=1) + self.eps
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            updates = torch.einsum('bjd,bij->bid', v, attention)
            
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            
            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(relu(self.fc1(self.norm_pre_ff(slots))))
            
        return slots

class SlotAttentionEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hidden_dim, device):
        super().__init__()
        self.device = device
        
        self.encoder = Encoder(resolution, hidden_dim)
        self.decoder = Decoder(hidden_dim, resolution)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.slot_attention = SlotAttention(num_slots, hidden_dim, num_iterations)
        
    def forward(self, _x):
        x = self.encoder(_x)
        x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        
        slots = self.slot_attention(x).reshape(-1, slots.shape[-1]).unsqueeze(1).unsqueeze(2).repeat((1, 8, 8, 1))
        
        x = self.decoder(slots)
        
        recons, masks = x.reshape(_x.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3, 1], dim=-1)
        
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1).permute(0, 3, 1, 2)
        
        return recon_combined, recons, masks, slots

class Trainer:
    def __init__(self, train_loader, device, model, learning_rate):
        self.train_loader = train_loader
        
        self.device = device
        
        self.model = model
        
        self.creterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train_step(self):
        loss_sum = 0
        for x, _ in tqdm(self.train_loader, desc='Training'):
            x = x.to(self.device)
            recon_combined, recons, masks, slots = self.model(x)
            
            loss = self.creterion(recon_combined, x)
            loss_sum += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        tqdm.write(f'Loss: {loss_sum / len(self.train_loader)}')
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            self.train_step()

if __name__ == '__main__':
    device = 'mps' if torch.backend.mps.is_available() else 'cpu'
    
    batch_size     = 32
    num_slots      = 8
    num_iterations = 3
    hidden_dim     = 64
    learning_rate  = 0.001
    num_epochs     = 1000
    resolution     = (128, 128)
    
    model = SlotAttentionEncoder(resolution, num_slots, num_iterations, hidden_dim, device).to(device)
    
    trainer = Trainer(train_loader, device, model, learning_rate)
    trainer.train(num_epochs)
    