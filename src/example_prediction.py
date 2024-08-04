import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2.functional import to_dtype
from animal_faces import AnimalFaces
from neural_network import NeuralNetwork
from random import randrange


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: \033[32m{device.upper()}\033[0m")

    test_dataset = AnimalFaces("data/afhq", train = False, device = device)
    index = randrange(0, len(test_dataset))

    image_tensor, label_tensor = test_dataset[index]
    image = to_pil_image(image_tensor.mul(255).to(dtype=torch.uint8))

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

    image_tensor = image_tensor.reshape(1, *image_tensor.shape)
    prediction = model(image_tensor)
    predicted_label = test_dataset.get_label(prediction.argmax(1).item()).capitalize()
    print(f"Predicted animal: {predicted_label}")
    print(f"Actual animal: {test_dataset.get_label(label_tensor.argmax(0).item()).capitalize()}")
    plt.imshow(image)
    plt.title(predicted_label)
    plt.show()

