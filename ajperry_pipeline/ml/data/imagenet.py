import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class ImageNetDataset(Dataset):
    def __init__(self, root: Path, data_partition = "train", desired_len: int = -1, rotation_degrees=5):
        image_folders = list((root / data_partition).glob("*"))
        self.classes = [line.strip() for line in (root / "wnids.txt").open("r").readlines()]
        self.class_defs = {line.split("\t")[0]:line.split("\t")[1].strip() for line in (root / "words.txt").open("r").readlines()}
        data = []
        self.is_train = data_partition=="train"
        self.rotation_degrees = rotation_degrees
        for image_folder in image_folders:
            class_name = image_folder.name
            assert class_name in self.class_defs, f"{class_name} not in {list(self.class_defs.keys())}"
            label_file = list(image_folder.glob("*.txt"))[0]
            class_index = self.classes.index(class_name)
            class_name_human = self.class_defs[class_name]
            for line in label_file.open().readlines():
                image_file, x_min, y_min, x_max, y_max  = line.split("\t")
                y_max =  y_max.strip()
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                image_file = image_folder / "images" / image_file
                data.append((image_file, x_min, y_min, x_max, y_max, class_index, class_name_human))
        self._root = root
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                
                transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.], std=[0.229, 0.224, 0.225, 1e-16])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.], std=[0.229, 0.224, 0.225, 1e-16])
            ])
        self.length = min(desired_len, len(data)) if desired_len!=-1 else len(data)
        self.data = data[:self.length]
        

    def __getitem__(self, idx):
        image_file, x_min, y_min, x_max, y_max, class_index, class_name_human = self.data[idx]
        im = Image.open(image_file).convert("RGBA")
        im = self.transform(im)
        return im, class_index

    def __len__(self):
        return self.length