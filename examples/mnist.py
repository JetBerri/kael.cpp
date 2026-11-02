import argparse
import time

import kael
from kael import nn, optim
from kael.data import DataLoader, TensorDataset, load_mnist
from kael.nn import functional as F


def accuracy(model, dataset, device):
    model.eval()
    correct = 0
    with kael.no_grad():
        for images, labels in DataLoader(dataset, batch_size=1000):
            predictions = kael.argmax(model(images.to(device)), 1)
            correct += (predictions == labels.to(device)).sum().item()
    return correct / len(dataset)


def train_epoch(model, optimizer, loader, device, epoch):
    model.train()
    for step, (images, labels) in enumerate(loader):
        loss = F.cross_entropy(model(images.to(device)), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    arguments = parser.parse_args()

    kael.manual_seed(42)
    train = TensorDataset(*load_mnist(arguments.data, train=True))
    test = TensorDataset(*load_mnist(arguments.data, train=False))
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, arguments.hidden),
        nn.ReLU(),
        nn.Linear(arguments.hidden, 10),
    ).to(arguments.device)
    optimizer = optim.AdamW(model.parameters())
    loader = DataLoader(train, batch_size=64, shuffle=True)

    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        train_epoch(model, optimizer, loader, arguments.device, epoch)
        elapsed = time.perf_counter() - started
        score = accuracy(model, test, arguments.device) * 100
        print(f"epoch {epoch} test accuracy {score:.2f}% in {elapsed:.2f} s")


if __name__ == "__main__":
    main()
