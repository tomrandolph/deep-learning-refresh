import torch
import torchvision
import tqdm
from torch import nn
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.model(x)


def train(
    model: "nn.Module",
    train_loader: "DataLoader",
    test_loader: "DataLoader",
    num_epochs: int = 5,
):
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for image, label in tqdm.tqdm(train_loader):
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {correct / total * 100:.2f}%"
        )

    print("Training complete.")


if __name__ == "__main__":
    mnist_training_dataset = torchvision.datasets.MNIST(
        root="./datasets",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )

    mnist_test_dataset = torchvision.datasets.MNIST(
        root="./datasets",
        download=True,
        train=False,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )

    mnist_training_dataloader = DataLoader(mnist_training_dataset)

    mnist_test_dataloader = DataLoader(mnist_test_dataset)

    device = torch.device("mps")  # on mac

    print("using device", device)
    model = Classifier()
    train(model, mnist_training_dataloader, mnist_test_dataloader, num_epochs=2)

    torch.save(model.state_dict(), "checkpoints/mnist_cnn.pt")
