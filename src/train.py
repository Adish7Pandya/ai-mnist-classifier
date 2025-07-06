import torch
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, train_loader, test_loader, epochs=5, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Training loss = {loss.item():.4f}")
    torch.save(model.state_dict(), "models/mnist_cnn.pth")
