from torch.utils.data import Dataset
from torch import tensor
from torchvision.transforms.v2.functional import to_dtype
import os
import torchvision.io
import torch


class AnimalFaces(Dataset):
    def __init__(self, directory: str, train: bool = True, transform = None, target_transform = None, device: str = "cuda") -> None:
        self._image_directory: str = directory + ("/train/" if train else "/val/")
        self._transform = transform
        self._target_transform = target_transform
        self._animal_map = {"cat": 0, "dog": 1, "wild": 2}
        self._animals = ["cat", "dog", "wild"]
        self._file_names = []
        list(map(self._file_names.extend, [[(animal, name) for name in os.listdir(self._image_directory + animal)] for animal in self._animals]))
        self._device = device


    def __len__(self):
        return len(self._file_names)
    

    def __getitem__(self, index):
        animal, file_name = self._file_names[index]
        image = to_dtype(torchvision.io.read_image(self._image_directory + f"{animal}/{file_name}"), dtype=torch.float32).mul(1.0 / 255).to(device=self._device)

        if self._transform is not None:
            image = self._transform(image)

        animal_index = self._animal_map[animal]
        animal_tensor = tensor([1 if i == animal_index else 0 for i in range(len(self._animals))], dtype=torch.float32, device=self._device)
        if self._target_transform is not None:
            animal_tensor = self._target_transform(animal_tensor)

        return image, animal_tensor


    def get_label(self, label_index):
        return self._animals[label_index]
