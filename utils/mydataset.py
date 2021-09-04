import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2


def transforms_(args, flag_train, img):
    """
    @author: ChenDong
    transforms function for training or testing.
    :param args: parameters set.
    :param flag_train: training flag --> Bool
    :return: 1 or 2 transforms function will be return(1 for train; 2 for test)
    """
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = (int(args.input_size), int(args.input_size))
    crop_size = (int(args.crop_size), int(args.crop_size))

    if flag_train:  # for train
        tsfm_train = [transforms.Resize(input_size), transforms.RandomCrop(crop_size),
                      transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean_vals, std_vals)]
        tc_train = transforms.Compose(tsfm_train)
        img_norm = tc_train(img)
        return img_norm

    else:  # for test
        if args.tencrop == 'True':
            func_transforms = [transforms.Resize(input_size),
                               transforms.TenCrop(crop_size),
                               transforms.Lambda(
                                   lambda crops: torch.stack(
                                       [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop
                                        in crops])),
                               ]
        else:
            func_transforms = [transforms.Resize(crop_size),
                               transforms.CenterCrop(crop_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean_vals, std_vals), ]
        tsfm_clstest = transforms.Compose(func_transforms)
        img_cls = tsfm_clstest(img)

        # transformation for test loc set
        tsfm_loctest = transforms.Compose([transforms.Resize(crop_size),  # 224
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean_vals, std_vals)])
        img_loc = tsfm_loctest(img.copy())
        return img_cls, img_loc


class dataset(Dataset):
    def __init__(self, args, train_flag, onehot_label=False, blur=None):
        self.args = args
        self.train = train_flag
        self.onehot_label = onehot_label
        self.blur = blur

        self.root_dir = args.img_dir
        self.datalist_file = args.train_list if self.train else args.test_list
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file, args.dataset)
        self.num_classes = args.num_classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.blur is not None:
            img = np.asarray(image)
            image = self.blur.augment_image(img)
            image = Image.fromarray(image)
            self.save_img(image, img_name)
        if self.onehot_label:
            gt_label = np.zeros(self.num_classes, dtype=np.float32)
            gt_label[self.label_list[idx].astype(int)] = 1
        else:
            gt_label = self.label_list[idx].astype(np.float32)

        return img_name, transforms_(self.args, self.train, image), gt_label

    def save_img(self, image, img_path, save_dir='./'):
        img_name = img_path.split('/')[-1]
        save_dir = os.path.join(save_dir, img_name)
        image.save(save_dir)

    def read_labeled_image_list(self, data_dir, data_list, dataset):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        if dataset == 'cub':
                            image += '.jpg'
                        elif dataset == 'ilsvrc':
                            image += '.JPEG'
                        else:
                            print('Wrong dataset.')
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels


class DataSetILSVRC(Dataset):
    def __init__(self,
                 datalist,
                 root_dir, transform=None,
                 with_path=False,
                 onehot_label=False,
                 num_classes=1000,
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file = datalist
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform
        self.onehot_label = onehot_label
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        gt_label = self.label_list[idx].astype(np.float32)

        if self.with_path:
            return img_name, image, gt_label
        else:
            return image, gt_label

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.JPEG'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels


# class PascalVOC(Dataset)


def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id


class dataset_with_mask(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, mask_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.with_path = with_path
        self.datalist_file = datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        mask_name = os.path.join(self.mask_dir, get_name_id(self.image_list[idx]) + '.png')
        mask = cv2.imread(mask_name)
        mask[mask == 0] = 255
        mask = mask - 1
        mask[mask == 254] = 255

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, mask, self.label_list[idx]
        else:
            return image, mask, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)
