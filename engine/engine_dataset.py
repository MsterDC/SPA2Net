import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


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
    def __init__(self, args, train_flag):
        self.args = args
        self.train = train_flag
        self.onehot_label = False
        self.blur = None
        self.datalist_file = args.train_list if self.train else args.test_list  # test.txt
        self.image_list = None
        self.label_list = None
        self.read_labeled_image_list(args.img_dir, self.datalist_file, args.dataset)
        if not train_flag:
            self.boxes_list = self.get_bbox()
        self.num_classes = args.num_classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')
        image_size = list(image.size)

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
        if self.train:
            return img_name, transforms_(self.args, self.train, image), gt_label
        else:
            bbox = self.boxes_list[idx]
            [x1, y1, x2, y2] = np.split(bbox, 4, 1)
            resize_size = self.args.crop_size
            crop_size = self.args.crop_size
            shift_size = 0
            [image_width, image_height] = image_size
            left_top_x = np.maximum(x1 / image_width * resize_size - shift_size, 0).astype(int)
            left_top_y = np.maximum(y1 / image_height * resize_size - shift_size, 0).astype(int)
            right_bottom_x = np.minimum(x2 / image_width * resize_size - shift_size, crop_size - 1).astype(int)
            right_bottom_y = np.minimum(y2 / image_height * resize_size - shift_size, crop_size - 1).astype(int)
            gt_bbox = np.concatenate((left_top_x, left_top_y, right_bottom_x, right_bottom_y), axis=1).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            return img_name, transforms_(self.args, self.train, image), gt_label, gt_bbox

    def save_img(self, image, img_path, save_dir='./'):
        img_name = img_path.split('/')[-1]
        save_dir = os.path.join(save_dir, img_name)
        image.save(save_dir)

    def get_bbox(self):
        gt_boxes = []
        with open(self.args.test_box, 'r') as f:
            for x in f.readlines():
                if self.args.dataset == 'ilsvrc':
                    x = x.strip().split(' ')[1:]
                elif self.args.dataset == 'cub':
                    x = x.strip().split(' ')[2:]
                if len(x) % 4 == 0:
                    gt_boxes.append(np.array(list(map(float, x))).reshape(-1, 4))
                else:
                    print('Wrong gt bboxes.')
        if self.args.dataset == 'cub':
            gt_boxes = [np.array([box[0][0], box[0][1], box[0][0] + box[0][2] - 1, box[0][1] + box[0][3] - 1]).reshape(-1, 4) for box in gt_boxes]
        return gt_boxes

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
        self.image_list, self.label_list = img_name_list, img_labels
        pass

