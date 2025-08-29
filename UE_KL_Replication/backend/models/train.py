import torch
from torch.utils.data import DataLoader, TensorDataset
from backend.models.model import SimpleUEModel
from backend.metrics.kl_divergence import kl_divergence

# Example synthetic dataset
X = torch.randn(100, 10)  # 100 samples, 10 features
Y_target = torch.randn(100, 5)  # target distributions over 5 classes

dataset = TensorDataset(X, Y_target)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
model = SimpleUEModel(input_dim=10, hidden_dim=32, output_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
def train_model():
    for epoch in range(5):  # 5 epochs for demo
        epoch_kl = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = kl_divergence(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_kl += loss.item()
        print(f"Epoch {epoch+1} KL Divergence: {epoch_kl/len(loader):.4f}")
