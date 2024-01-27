import torchvision
import matplotlib.pyplot as plt
import numpy as np

mnist_dataset = torchvision.datasets.MNIST(
    root="./datasets",
    download=True,
    train=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)

idx = np.random.randint(0, len(mnist_dataset))

digit, label = mnist_dataset[idx]

plt.imshow(np.squeeze(digit))
plt.title(label)
plt.show()
