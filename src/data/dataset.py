import torch
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np


class McJVSDataset(Dataset):
    def __init__(self, root="36_40_melceps", dis=[0, 1, 2, 3]):
        super().__init__()

        self.fs = 24000
        self.dis = dis
        self.data = list(self.extract_data(root))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def extract_data(self, root):
        for i in os.listdir(root):
            data = torch.load(os.path.join(root, i))
            if not torch.argmax(data[1]) in self.dis:
                yield data


class ImageDataset(Dataset):
    def __init__(self, root="images"):
        super().__init__()
        self.root = root
        self.data = self._extract_data()

    def __getitem__(self, index):
        return self._get_random(index, 1)

    def _get_random(self, index, label):

        return np.random.choice(self.data[label][index])

    def __len__(self):
        return len(self.data)

    def _extract_data(self):
        data = [[[] for i in range(51)] for j in range(3)]
        with open(self.root+"/"+"crossdata.txt") as f:
            for i in f:
                index, image, label = i.lstrip().split()
                image = torchvision.transforms.Resize((32, 32))(torchvision.io.read_image(self.root+"/"+image))
                image = image.to(torch.float)
                data[int(label)][int(index)].append(image)
        return data


class CrossDataset(Dataset):
    def __init__(self, audio_data_dir="36_40_melceps", image_data_dir="images"):
        super().__init__()
        self.image = ImageDataset(root=image_data_dir)
        self.audio = McJVSDataset(root=audio_data_dir)

    def __getitem__(self, index):
        audio, label = self.audio[index]
        image = self.image[torch.argmax(label)]

        return image, audio, label

    def __len__(self):
        return self.audio.__len__()
