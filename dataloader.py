import torch.utils.data as data
from PIL import Image
import torchvision.transforms as tt
import os

def load_img(filepath):
    return Image.open(filepath).convert("RGB")

class TrainDataset(data.Dataset):
    def __init__(self, inputset, labelset):
        super(TrainDataset, self).__init__()
        self.input_path = 'dataset/' + inputset
        self.label_path = 'dataset/' + labelset
        self.input_name = []
        self.label_name = []

        for img_name in os.listdir(self.input_path):
            self.input_name.append(img_name)
        for img_name in os.listdir(self.label_path):
            self.label_name.append(img_name)

        self.input_len = len(self.input_name)
        self.label_len = len(self.label_name)
        print('input image count: ', self.input_len)
        print('label image count: ', self.label_len)

    def __getitem__(self, index):
        input = os.path.join(self.input_path, self.input_name[index % self.input_len])
        label = os.path.join(self.label_path, self.label_name[index % self.label_len])
        input_img = tt.ToTensor()(load_img(input))
        label_img = tt.ToTensor()(load_img(label))
        input_img = tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_img)
        label_img = tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(label_img)

        return input_img, label_img

    def __len__(self):
        if self.input_len > self.label_len:
            return self.input_len
        else:
            return self.label_len

class TestDataset(data.Dataset):
    def __init__(self, inputset):
        super(TestDataset, self).__init__()
        self.input_path = 'dataset/' + inputset
        self.input_name = []

        for img_name in os.listdir(self.input_path):
            self.input_name.append(img_name)

        self.input_len = len(self.input_name)
        print('test image count: ', self.input_len)

    def __getitem__(self, index):
        input = os.path.join(self.input_path, self.input_name[index])
        input_img = tt.ToTensor()(load_img(input))
        return input_img

    def __len__(self):
        return self.input_len
