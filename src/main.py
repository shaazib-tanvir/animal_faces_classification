import torch
import os.path
from torch.utils.data import DataLoader
from animal_faces import AnimalFaces
from neural_network import NeuralNetwork

def train(dataloader: DataLoader, model: NeuralNetwork, lossfn, optimizer):
    model.train()

    for batch, (x, y) in enumerate( dataloader):
        prediction = model(x)
        loss = lossfn(prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 50 == 0:
            loss = loss.item()
            print(f"Loss: {loss}")

    torch.save(model.state_dict(), "model.pth")
    print(f"Saved model")


def test(dataloader, model: NeuralNetwork):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            prediction = model(x)
            correct += (prediction.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    print(f"Accuracy: \033[32m{100 * correct / size}%\033[0m")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: \033[32m{device.upper()}\033[0m")

    train_dataset = AnimalFaces("data/afhq", train = True, device = device)
    test_dataset = AnimalFaces("data/afhq", train = False, device = device)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle = True, batch_size = batch_size)

    model = NeuralNetwork().to(device=device)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    lossfn = torch.nn.MSELoss()
    epochs = 5

    for epoch in range(epochs):
        print(f"\033[34mEpoch\033[0m {epoch}")
        print("="*20)
        train(train_dataloader, model, lossfn, optimizer)
        test(test_dataloader, model)
        print("="*20)
