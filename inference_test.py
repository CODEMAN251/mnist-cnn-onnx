import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load MNIST test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_test = datasets.MNIST(root='.', train=False, download=True, transform=transform)

# Take one sample
image, label = mnist_test[0]
input_tensor = image.unsqueeze(0).numpy()

# Load ONNX model
session = ort.InferenceSession("mnist_cnn.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor})[0]

# Get predicted digit
pred = np.argmax(output)

# Show result
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Prediction: {pred} (True label: {label})")
plt.axis("off")
plt.show()
