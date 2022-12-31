import h5py

import os
from glob import glob
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageH5Dataset(Dataset):

    def __init__(self, dataset_path, img_dim):
        self.dataset_path = dataset_path
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.live_data = None
        self.images = None

        with h5py.File(self.dataset_path, 'r') as file:
            self.dataset_len = len(file["image"])

    def __getitem__(self, idx):
        if self.live_data is None:
            self.live_data = h5py.File(self.dataset_path, 'r')
            self.images = self.live_data['image']

        image = self.images[idx]

        h, w, c = image.shape
        top_h = int((h - w) / 2)

        image = image[top_h:top_h + w]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return self.dataset_len


class ImageFolderDataset(Dataset):

    def __init__(self, image_dir, img_dim):
        self.img_paths = glob(os.path.join(image_dir, "*.jpg"))
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return len(self.img_paths)

class ImageAttrDataset(Dataset):

    def __init__(self, image_dir, img_dim, attrs):
        #self.img_paths = []
        self.attrs = attrs

        #print(image_dir)
        # for i in image_dir:
        #     print(i)
        self.img_paths = glob(os.path.join(image_dir, "*.jpg"))
        #print(self.img_paths)
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        # self.img_attr = pd.read_csv('data/celeba/Anno/list_attr_celeba.txt', delim_whitespace=True,
        #                    usecols=['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        #                             'Wearing_Hat', 'Young'])

        #这里是把数据集里的所有图片的三个特征及其对应值都拿出来了
        self.img_attr = pd.read_csv('data/celeba/Anno/list_attr_celeba.csv', sep=',', dtype=int, usecols=attrs)
        # print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        # print(self.img_attr)

    #返回一条训练数据，并将其转化为tensor
    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #拿出这个客户端数据集的所有图片的名字
        img_name = self.img_paths[index].split('/')[-1][:-4]
        img_name = img_name.replace('0\\','')
        #print(img_name)

        # label = self.img_attr.loc[img_name+'.jpg'].to_dict('records')
        # print(self.img_attr.loc[int(img_name)-1])

        labels = self.img_attr.loc[int(img_name)-1].to_dict()
        #print(labels)  结果如下，这里输出的是一个字典
        #{'Blond_Hair': 1, 'Male': -1, 'Narrow_Eyes': -1}

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.
        #print(labels[self.attrs[0]])
        #return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), label
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), labels

    def __len__(self):
        return len(self.img_paths)

class CelebaDataset(Dataset):

    def __init__(self, image_dir, img_dim, attrs):
        self.img_paths = []
        self.attrs = attrs
        self.img_paths.extend(glob(os.path.join(image_dir, "*.png")))
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        # self.img_attr = pd.read_csv('data/celeba/Anno/list_attr_celeba.txt', delim_whitespace=True,
        #                    usecols=['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        #                             'Wearing_Hat', 'Young'])

        self.img_attr = pd.read_csv('data/celeba/Anno/list_attr_celeba.csv', sep=',', dtype=int, usecols=attrs)
        # print(self.img_attr)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_name = self.img_paths[idx].split('/')[-1][:-4]
        # label = self.img_attr.loc[img_name+'.jpg'].to_dict('records')
        # print(self.img_attr.loc[int(img_name)-1])

        label = self.img_attr.loc[int(img_name)-1].to_dict()

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        # return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), label
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), 1 if label[self.attrs[0]] == 1 else 0

    def __len__(self):
        return len(self.img_paths)

### cluster
class NISTDataset(Dataset):

    def __init__(self, image_dir, img_dim, group='hsf_0', transform=None):
        self.img_paths = []
        for i in range(26):
            png_path = os.path.join(image_dir, str(i), group)
            self.img_paths.extend(glob(os.path.join(png_path, "*.png")))

        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.img_paths[idx].split('/')[-3])

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        tensor_img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return tensor_img, label

    def __len__(self):
        return len(self.img_paths)


class NISTClusterDataset(Dataset):

    def __init__(self, image_dir, img_dim, group='hsf_0', class_idx=0, transform=None):
        self.img_paths = []
        if type(class_idx) is list:
            for i in class_idx:
                png_path = os.path.join(image_dir, str(i), group)
                self.img_paths.extend(glob(os.path.join(png_path, "*.png")))
        else:
            png_path = os.path.join(image_dir, str(class_idx), group)
            self.img_paths.extend(glob(os.path.join(png_path, "*.png")))

        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.transform = transform

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        tensor_img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return self.transform(tensor_img), path

    def __len__(self):
        return len(self.img_paths)


class CelebaImageName(CelebaDataset):
    def __init__(self, image_dir, img_dim, attrs):
        super().__init__(image_dir, img_dim, attrs)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_name = self.img_paths[idx].split('/')[-1][:-4]
        # label = self.img_attr.loc[img_name+'.jpg'].to_dict('records')
        # print(self.img_attr.loc[int(img_name)-1])

        label = self.img_attr.loc[int(img_name)-1].to_dict()
        label = 1 if label[self.attrs[0]] == 1 else 0

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), label, img_name


class CelebaImageMask(Dataset):
    def __init__(self, image_dir, mask_dir, img_dim):
        self.img_paths = []
        self.mask_path = mask_dir
        self.img_paths.extend(glob(os.path.join(image_dir, "*.png")))
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

    def __getitem__(self, idx):

        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_name = self.img_paths[idx].split('/')[-1][:-4]
        mask = np.load(os.path.join(self.mask_path, img_name+'.npy'))

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.unsqueeze(torch.FloatTensor(1.0-mask), 0)

    def __len__(self):
        return len(self.img_paths)