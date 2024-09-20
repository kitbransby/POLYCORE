import pathlib
import re
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import os
import cv2

def load_dataset(config):

    # TODO update paths
    train_path = config['DIR'] + "Dataset/" + config['DATASET_NAME'] + "/Train"
    test_path = config['DIR'] + "Dataset/" + config['DATASET_NAME'] + "/Test"

    transforms_train = [A.geometric.rotate.Rotate(180, p=1, interpolation=2, border_mode=1),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15)]

    img_path = os.path.join(train_path, config['IMAGE_INP'])
    mask_path = os.path.join(train_path, 'Mask_480')

    train_dataset = IVUSDataset(img_path=img_path,
                                mask_path=mask_path,
                                transform=A.Compose(transforms_train),
                                preprocessing=config['PREPROCESSING'],
                                mask_size=config['MASK_SIZE'],
                                resolution=config['RESOLUTION'],
                                norm=config['INPUT_NORM'])

    img_path = os.path.join(test_path, config['IMAGE_INP'])
    mask_path = os.path.join(test_path, 'Mask_480')

    test_dataset = IVUSDataset(img_path=img_path,
                               mask_path=mask_path,
                               transform=False,
                               preprocessing=config['PREPROCESSING'],
                               mask_size=config['MASK_SIZE'],
                               resolution=config['RESOLUTION'],
                               norm=config['INPUT_NORM'])

    return train_dataset, test_dataset

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class IVUSDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_path, mask_path, transform=None, preprocessing=None, mask_size=480, resolution=480, norm='z'):

        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.preprocessing = preprocessing
        self.mask_size = mask_size
        self.resolution = resolution
        self.norm = norm

        self.images = self.load_file_list(img_path)
        self.masks = self.load_file_list(mask_path)

        print('Dataset: {} Images and {} Masks'.format(len(self.images), len(self.masks)))

    def load_file_list(self, path):
        if isinstance(path, list) and len(path) > 1:
            all_files = []
            for i in range(len(path)):
                data_root = pathlib.Path(path[i])
                all_files += list(data_root.glob('*.npy'))
        else:
            data_root = pathlib.Path(path)
            all_files = list(data_root.glob('*.npy'))
        all_files = [str(path) for path in all_files]
        all_files.sort()
        return all_files


    def __len__(self):
        return len(self.images)

    def get_bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array([cmin, rmin, cmax, rmax])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # LOAD DATA
        img_name = self.images[idx]
        image = np.load(img_name)
        mask_name = self.masks[idx]
        mask = np.load(mask_name)
        id = img_name.split('/')[-1].split('.')[0]

        # AUGMENTATION
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # GENERATE BBOX FROM MASK
        eem_bbox = self.get_bbox(mask.astype(np.int32) != 0)
        lumen_bbox = self.get_bbox(mask.astype(np.int32) == 255)
        bbox = np.concatenate([eem_bbox, lumen_bbox])
        bbox = np.clip(bbox, 0, 480)
        bbox = torch.from_numpy(bbox).float() / 480

        # FORMAT IMAGE
        if self.resolution != 480:
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose((2,0,1))
        image = np.clip(image, 0, 255)
        if self.norm == '01':
            image = image / 255
        elif self.norm == 'z':
            # TODO: fix this bug. should be divided by dataset mean and var
            image = (image - image.mean()) / np.sqrt(image.var())
        elif self.norm == 'none':
            image = image
        else:
            print('no input norm selected')
        image = torch.from_numpy(image).float()
        if self.preprocessing == 'NONE':
            image = image[0,:,:].unsqueeze(0)

        # FORMAT MASK
        mask[mask == 128] = 1
        mask[mask == 255] = 2
        if self.mask_size != 480:
            mask = cv2.resize(mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_AREA)
        mask = torch.from_numpy(mask).float()

        return {'image': image, 'mask': mask, 'id': id, 'bbox': bbox}


