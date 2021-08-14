import torchvision.io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from bacteria_dataset import BacteriaDataset

if __name__ == "__main__":

    if os.path.exists("data/refined"):
        print("Data exists")
        exit(0)

    os.mkdir("data/refined")
    os.mkdir("data/refined/images")
    os.mkdir("data/refined/masks")

    ds = BacteriaDataset(is_train=True)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    min_height = 10000
    min_width = 10000
    for image, mask, name in dataloader:
        if image.shape != mask.shape:
            print("Shape not same")
        if image.shape[2] < min_height:
            min_height = image.shape[2]

        if image.shape[3] < min_width:
            min_width = image.shape[3]

    resizer = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((min_height, min_width)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    for image, mask, name in dataloader:
        image = resizer(image)
        mask = resizer(mask)
        torchvision.utils.save_image(image, f"data/refined/images/{name}" )
        torchvision.utils.save_image(mask, f"data/refined/masks/{name}" )