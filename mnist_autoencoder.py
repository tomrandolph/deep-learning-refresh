import math
from typing import Sequence

import torch
import torchvision
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, image_size: tuple[int, int, int], embedding_size: int) -> None:
        super(Encoder, self).__init__()
        h_in, w_in, channels_in = image_size
        self.channels = 32, 64
        self.shape_before_flatten = self.channels[-1], (h_in // 4), (w_in // 4)
        self.model = nn.Sequential(
            nn.Conv2d(
                channels_in, self.channels[0], kernel_size=3, stride=2, padding=1
            ),  # N, 1, h, w -> N, 32, h/2, h/2
            nn.ReLU(),  # N, 32, h/2, w/2 -> N, 32, h/2, w/2
            nn.Conv2d(
                self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        self.fc = nn.Linear(math.prod(self.shape_before_flatten), embedding_size)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.model(x)
        x = x.view(x.size(0), math.prod(self.shape_before_flatten))

        return self.fc(x)


class Decoder(nn.Module):
    def __init__(
        self,
        shape_before_flatten: tuple[int, int, int],
        channels: Sequence[int],
        image_size: tuple[int, int, int],
        embedding_size: int,
    ) -> None:
        super(Decoder, self).__init__()
        h_out, w_out, channels_out = image_size

        self.shape_before_flatten = shape_before_flatten
        channels = tuple(channels[::-1])
        self.fc = nn.Linear(embedding_size, math.prod(shape_before_flatten))

        self.conv1 = nn.ConvTranspose2d(
            channels[0],
            channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.conv2 = nn.ConvTranspose2d(
            channels[1],
            channels_out,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        #     self.model = nn.Sequential(
        # nn.Conv2d(
        #     channels_in, self.channels[1], kernel_size=3, stride=2, padding=1
        # ),  # N, 1, h, w -> N, 32, h/2, h/2
        # nn.ReLU(),  # N, 32, h/2, w/2 -> N, 32, h/2, w/2
        # nn.MaxPool2d(2, 2),  # N, 32, h/2, w/2 -> N, 32, h/4, h/4
        # nn.Conv2d(
        #     self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=1
        # ),

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.fc(x)
        x = x.view(x.size(0), *self.shape_before_flatten)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.sigmoid(self.conv2(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, image_size: tuple[int, int, int], embedding_size: int) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(image_size, embedding_size)
        self.decoder = Decoder(
            shape_before_flatten=self.encoder.shape_before_flatten,
            channels=self.encoder.channels,
            image_size=image_size,
            embedding_size=embedding_size,
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.encoder(x)
        return self.decoder(x)


model = AutoEncoder((28, 28, 1), 120)

mnist_training_dataset = torchvision.datasets.MNIST(
    root="./datasets",
    download=True,
    train=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)
mnist_training_dataloader = DataLoader(mnist_training_dataset, batch_size=16)


mnist_test_dataset = torchvision.datasets.MNIST(
    root="./datasets",
    download=True,
    train=False,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)

mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=8)


def plot_images(images1, images2):
    assert len(images1) == len(images2), "The two sequences must have the same length."

    fig, axs = plt.subplots(2, len(images1), figsize=(15, 6))

    for i, (img1, img2) in enumerate(zip(images1, images2)):
        img1 = img1.cpu().numpy().squeeze()
        img2 = img2.cpu().numpy().squeeze()

        axs[0, i].imshow(img1, cmap="gray")
        axs[0, i].axis("off")

        axs[1, i].imshow(img2, cmap="gray")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def train(
    model: "nn.Module",
    train_loader: "DataLoader",
    test_loader: "DataLoader",
    num_epochs: int = 5,
    device="mps",
):
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for image, _ in tqdm.tqdm(train_loader):
            image = image.to(device)

            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, image)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                if i > 2:
                    break
                image = images[0]
                images = images.to(device)
                outputs = model(images)
                plot_images(images, outputs)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete.")


train(model, mnist_training_dataloader, mnist_test_dataloader)
