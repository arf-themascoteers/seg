import torchvision.io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


class BacteriaDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        files = os.listdir("data/images")
        self.names = None
        if is_train:
            self.names = [files[i] for i in range(len(files)) if i % 3 != 0]
        else:
            self.names = [files[i] for i in range(len(files)) if i % 3 == 0]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(f"data/images/{self.names[idx]}")
        mask = torchvision.io.read_image(f"data/masks/{self.names[idx]}")
        return image, mask, self.names[idx]

if __name__ == "__main__":
    ds = BacteriaDataset(is_train=True)
    dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    min_height = 10000
    min_width = 10000
    for image, mask in dataloader:
        if image.shape != mask.shape:
            print("Shape not same")
        if image.shape[2] < min_width:
            min_width = image.shape[2]

        if image.shape[3] < min_height:
            min_height = image.shape[3]

        print(min_height, min_width)