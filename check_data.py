from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt

dataset = PneumoniaMNIST(
    split="train",
    download=True
)

image, label = dataset[0]

print("Image size:", image.size)
print("Label:", label)

plt.imshow(image, cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()