import os
import torch
import pandas as pd
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms

def build_transform(is_train, args, gen_attn=False):
    if is_train and not gen_attn:
        return transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor()
        ])

class BraTSDataset(Dataset):
    def __init__(self, csv_path, base_dir, config, train=True, transform=None, gen_attn=False):
        self.dataframe = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.config = config
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        # Grabbing the relative path from CSV and append to base_dir
        rel_img_name_flair = self.dataframe.iloc[index, 0]
        img_name_flair = os.path.join(self.base_dir, rel_img_name_flair)
        
        img_name_t1ce = img_name_flair.replace("flair", "t1ce")
        img_name_t2 = img_name_flair.replace("flair", "t2")

        image_flair = Image.open(img_name_flair).convert("L")
        image_t1ce = Image.open(img_name_t1ce).convert("L")
        image_t2 = Image.open(img_name_t2).convert("L")

        image = Image.merge("RGB", (image_flair, image_t1ce, image_t2))
        return image

    def _process_labels(self, index):
        label_dict = self.dataframe.iloc[index, 3:].to_dict()
        combined_labels = {}
        for new_class, old_classes in self.config["combine"].items():
            combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))
        
        keys = sorted(self.config['combine'].keys())
        return torch.tensor([combined_labels[k] for k in keys], dtype=torch.float32)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self._load_image(index)
        label = self._process_labels(index)
        
        if self.transform:
            image = self.transform(image)

        return image, label


class BraTSDatasetMS(Dataset):
    def __init__(self, csv_path, base_dir, config, scales, train=True, transform=None, gen_attn=False, unit=1):
        self.dataframe = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.config = config
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn
        self.scales = scales
        self.unit = unit

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        rel_img_name_flair = self.dataframe.iloc[index, 0]
        img_name_flair = os.path.join(self.base_dir, rel_img_name_flair)
        
        img_name_t1ce = img_name_flair.replace("flair", "t1ce")
        img_name_t2 = img_name_flair.replace("flair", "t2")

        image_flair = Image.open(img_name_flair).convert("L")
        image_t1ce = Image.open(img_name_t1ce).convert("L")
        image_t2 = Image.open(img_name_t2).convert("L")

        image = Image.merge("RGB", (image_flair, image_t1ce, image_t2))
        return image

    def _process_labels(self, index):
        label_dict = self.dataframe.iloc[index, 3:].to_dict()
        combined_labels = {}
        for new_class, old_classes in self.config["combine"].items():
            combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))
        
        keys = sorted(self.config['combine'].keys())
        return torch.tensor([combined_labels[k] for k in keys], dtype=torch.float32)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = self._load_image(index)
        label = self._process_labels(index)

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s), round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
            
        return msf_img_list, label


def build_dataset(is_train, args, split='train'):
    # Determines if we are just generating attention maps (not training)
    gen_attn = (split != 'train')
    transform = build_transform(is_train, args, gen_attn)
    
    config = {
        'dataset': 'brats',
        'task': 'multiclass',
        'combine': {
            'core': ['necrosis', 'enhancing'],
            'edema': ['edema']
        }
    }
    nb_classes = len(config['combine'])
    
    # Dynamically select train.csv, val.csv, or test.csv based on the split argument
    csv_path = os.path.join(args.data_path, f'{split}.csv')

    if args.data_set == 'BRATS':
        dataset = BraTSDataset(csv_path=csv_path, base_dir=args.data_path, config=config, 
                               train=is_train, gen_attn=gen_attn, transform=transform)
    elif args.data_set == 'BRATSMS':
        dataset = BraTSDatasetMS(csv_path=csv_path, base_dir=args.data_path, config=config, scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform)
    else:
        raise ValueError(f"Unknown dataset mode: {args.data_set}")

    return dataset, nb_classes