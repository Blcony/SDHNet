import os
from os.path import join
import random
import numpy as np
import tifffile as tif
import SimpleITK as sitk

import torch
import torch.utils.data as data


def load_nii(path):
    nii = sitk.ReadImage(path)
    nii_array = sitk.GetArrayFromImage(nii)
    return nii_array


def read_datasets(path, datasets):
    files = []
    for d in datasets:
        files.extend([join(path, d, i) for i in os.listdir(join(path, d))])
    return files


def generate_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
    return pairs


def generate_pairs_val(files):
    pairs = []
    labels = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
                labels.append([join(d1, 'segmentation.tif'), join(d2, 'segmentation.tif')])
    return pairs, labels


def generate_lspig_val(files):
    pairs = []
    labels = []
    files.sort()
    for i in range(0, len(files), 2):
        d1 = files[i]
        d2 = files[i + 1]
        pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
        labels.append([join(d1, 'segmentation.tif'), join(d2, 'segmentation.tif')])
        pairs.append([join(d2, 'volume.tif'), join(d1, 'volume.tif')])
        labels.append([join(d2, 'segmentation.tif'), join(d1, 'segmentation.tif')])

    return pairs, labels


def generate_atlas(atlas, files):
    pairs = []
    for d in files:
        pairs.append([join(atlas, 'volume.tif'), join(d, 'volume.tif')])

    return pairs


def generate_atlas_val(atlas, files):
    pairs = []
    labels = []
    for d in files:
        if 'S01' in d:
            continue
        pairs.append([join(atlas, 'volume.tif'), join(d, 'volume.tif')])
        labels.append([join(atlas, 'segmentation.tif'), join(d, 'segmentation.tif')])
    return pairs, labels


def generate_oasis_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'aligned_norm.nii.gz'), join(d2, 'aligned_norm.nii.gz')])
    return pairs


def generate_oasis_pairs_val(files):
    pairs = []
    labels = []
    for i in range(len(files)-1):
        for j in range(len(files)-1):
            if i != j:
                pairs.append([join(files[i], 'aligned_norm.nii.gz'), join(files[i+1], 'aligned_norm.nii.gz')])
                labels.append([join(files[i], 'aligned_seg35.nii.gz'), join(files[i+1], 'aligned_seg35.nii.gz')])
    return pairs, labels


def generate_mindboggle_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'data.nii.gz'), join(d2, 'data.nii.gz')])
    return pairs


def generate_mindboggle_pairs_val(files):
    pairs = []
    labels = []
    for i in range(len(files) - 1):
        for j in range(len(files) - 1):
            if i != j:
                pairs.append([join(files[i], 'data.nii.gz'), join(files[i+1], 'data.nii.gz')])
                labels.append([join(files[i], 'seg.nii.gz'), join(files[i+1], 'seg.nii.gz')])
    return pairs, labels


class LiverTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [128, 128, 128]
        self.datasets = ['msd_hep', 'msd_pan', 'msd_liver', 'youyi_liver']
        self.train_path = join(args.data_path, 'Train')
        self.files = read_datasets(self.train_path, self.datasets)
        self.pairs = generate_pairs(self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        index = index % len(self.pairs)
        data1, data2 = self.pairs[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        return image1, image2

    def __len__(self):
        return len(self.pairs)


class LiverTest(data.Dataset):
    def __init__(self, args, datas):
        self.size = [128, 128, 128]
        self.datasets = [datas]
        self.test_path = join(args.data_path, 'Test')
        self.files = read_datasets(self.test_path, self.datasets)
        self.pairs, self.labels = generate_pairs_val(self.files)

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class LspigTest(data.Dataset):
    def __init__(self, args, datas):
        self.size = [128, 128, 128]
        self.datasets = [datas]
        self.test_path = join(args.data_path, 'Test')
        self.files = read_datasets(self.test_path, self.datasets)
        self.pairs, self.labels = generate_lspig_val(self.files)

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class BrainTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [128, 128, 128]
        self.datasets = ['abide', 'abidef', 'adhd', 'adni']
        self.train_path = join(args.data_path, 'Train')
        self.atlas = join(args.data_path, 'Test/lpba/S01')
        self.files = read_datasets(self.train_path, self.datasets)
        self.pairs = generate_atlas(self.atlas, self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        index = index % len(self.pairs)
        data1, data2 = self.pairs[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        return image1, image2

    def __len__(self):
        return len(self.pairs)


class BrainTest(data.Dataset):
    def __init__(self, args, datas):
        self.size = [128, 128, 128]
        self.datasets = [datas]
        self.test_path = join(args.data_path, 'Test')
        self.atlas = join(args.data_path, 'Test/lpba/S01')
        self.files = read_datasets(self.test_path, self.datasets)
        self.pairs, self.labels = generate_atlas_val(self.atlas, self.files)
        self.seg_values = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                           50, 61, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102,
                           121, 122, 161, 162, 163, 164, 165, 166, 181, 182]

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class OasisTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [160, 192, 224]
        self.datasets = ['oasis_train']
        self.files = read_datasets(args.data_path, self.datasets)
        self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        self.pairs = generate_oasis_pairs(self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        index = index % len(self.pairs)
        data1, data2 = self.pairs[index]

        image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()

        return image1, image2

    def __len__(self):
        return len(self.pairs)


class OasisTest(data.Dataset):
    def __init__(self, args):
        self.size = [160, 192, 224]
        self.datasets = ['oasis_val']
        self.files = read_datasets(args.data_path, self.datasets)
        self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        self.pairs, self.labels = generate_oasis_pairs_val(self.files)
        self.seg_values = [i for i in range(1, 36)]

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()

        label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class MindboggleTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [192, 192, 192]
        self.datasets = ['mindboggle_train']
        self.files = read_datasets(args.data_path, self.datasets)
        self.files = sorted(self.files, key=lambda x: int(x[-1]))
        self.pairs = generate_mindboggle_pairs(self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        index = index % len(self.pairs)
        data1, data2 = self.pairs[index]

        image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()

        return image1, image2

    def __len__(self):
        return len(self.pairs)


class MindboggleTest(data.Dataset):
    def __init__(self, args):
        self.size = [192, 192, 192]
        self.datasets = ['mindboggle_val']
        self.files = read_datasets(args.data_path, self.datasets)
        self.files = sorted(self.files, key=lambda x: int(x[-1]))
        self.pairs, self.labels = generate_mindboggle_pairs_val(self.files)
        self.pairs = self.pairs[5:]
        self.labels = self.labels[5:]
        self.seg_values = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 43, 44, 46, 47,
                           49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77, 80, 85, 251, 252, 253, 254, 255, 1000, 1002,
                           1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
                           1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                           2000, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
                           2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                           2034, 2035]

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()

        label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)

