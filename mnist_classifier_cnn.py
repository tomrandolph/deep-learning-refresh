import torch
import torchvision
import tqdm
from torch import nn
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1
        )  # N, 1, 28, 28
        self.pool = nn.MaxPool2d(2, 2)  # N, 32, 7, 7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


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

    mnist_training_dataloader = DataLoader(mnist_training_dataset, batch_size=8)

    mnist_test_dataloader = DataLoader(mnist_test_dataset)

    device = torch.device("mps")  # on mac

    print("using device", device)
    model = Classifier()
    train(model, mnist_training_dataloader, mnist_test_dataloader, num_epochs=3)

    torch.save(model.state_dict(), "checkpoints/mnist_cnn.pt")

    # device = torch.device("mps")  # on mac
    # model = Classifier()
    # model.load_state_dict(torch.load("checkpoints/mnist_cnn.pt"))
    # model.eval()
    # model = model.to(device)

    # # Make predictions

    # # Plot the images and their predicted labels
    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(4, 5, figsize=(15, 12))  # Create a grid of 4x5 subplots
    # axs = axs.ravel()  # Flatten the grid to easily iterate over it

    # for i, (image, label) in enumerate(iter(mnist_test_dataloader)):
    #     if i >= 20:
    #         break
    #     # Move images and labels to the device
    #     image = image.to(device)
    #     label = label.to(device)

    #     outputs = model(image)
    #     _, predicted = torch.max(outputs, 1)

    #     image = image.cpu().numpy().squeeze()  # Move image to cpu and convert to numpy
    #     axs[i].imshow(image, cmap="gray")
    #     axs[i].set_title(f"Predicted: {predicted.squeeze().item()}")
    #     axs[i].axis("off")  # Hide axis

    # plt.tight_layout()
    # plt.show()
