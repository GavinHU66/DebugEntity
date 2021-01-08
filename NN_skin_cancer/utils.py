import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
import math
from PIL import Image
import types
import pandas as pd
from tools.auto_augment import AutoAugment, Cutout


# Define ISIC Dataset Class
class ISICDataset(Dataset):
    """ISIC dataset."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # Mdlparams
        self.mdlParams = mdlParams
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indSet = indSet
        if self.indSet == 'trainInd':
            self.root = mdlParams['dataDir'] + '/train'
        elif self.indSet == 'valInd':
            self.root = mdlParams['dataDir'] + '/valid'
        else:
            self.root = mdlParams['dataDir'] + '/test'
        self.names_list = []
        # Number of classes
        self.numClasses = mdlParams['numClasses']
        # Size to crop
        self.crop_size = mdlParams['crop_size']
        # Model input size
        self.input_size = (np.int32(mdlParams['input_size'][0]), np.int32(mdlParams['input_size'][1]))
        # Potential class balancing option 
        self.balancing = mdlParams['balance_classes']
        # Potential setMean and setStd
        self.setMean = mdlParams['setMean'].astype(np.float32)
        self.setStd = mdlParams['setStd'].astype(np.float32)
        # Only downsample
        self.only_downsmaple = mdlParams.get('only_downsmaple', False)
        # Meta csv
        self.meta_path = mdlParams['meta_path']
        self.meta_df = pd.read_pickle(self.meta_path)

        class_label = 0
        self.subsets_size = []
        self.image_path = []
        for dir in os.listdir(self.root):
            subset_size = 0
            dir_path = os.path.join(self.root, dir)
            for image in os.listdir(dir_path):
                self.names_list.append({
                    'id': image.split('.')[0],
                    'label': class_label,
                })
                self.image_path.append(os.path.join(dir_path))
                subset_size += 1
            self.subsets_size.append(subset_size)
            class_label += 1

        if indSet == 'trainInd':
            all_transforms = []
            if self.only_downsmaple:
                all_transforms.append(transforms.Resize(self.input_size))
            else:
                all_transforms.append(transforms.Resize(self.crop_size))
                all_transforms.append(transforms.RandomCrop(self.input_size))
            if mdlParams.get('flip_lr_ud', False):
                all_transforms.append(transforms.RandomHorizontalFlip())
                all_transforms.append(transforms.RandomVerticalFlip())
            # Full rot
            if mdlParams.get('full_rot', 0) > 0:
                if mdlParams.get('scale', False):
                    all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(mdlParams['full_rot'],
                                                                                           scale=mdlParams['scale'],
                                                                                           shear=mdlParams.get('shear',
                                                                                                               0),
                                                                                           resample=Image.NEAREST),
                                                                   transforms.RandomAffine(mdlParams['full_rot'],
                                                                                           scale=mdlParams['scale'],
                                                                                           shear=mdlParams.get('shear',
                                                                                                               0),
                                                                                           resample=Image.BICUBIC),
                                                                   transforms.RandomAffine(mdlParams['full_rot'],
                                                                                           scale=mdlParams['scale'],
                                                                                           shear=mdlParams.get('shear',
                                                                                                               0),
                                                                                           resample=Image.BILINEAR)]))
                else:
                    all_transforms.append(transforms.RandomChoice(
                        [transforms.RandomRotation(mdlParams['full_rot'], resample=Image.NEAREST),
                         transforms.RandomRotation(mdlParams['full_rot'], resample=Image.BICUBIC),
                         transforms.RandomRotation(mdlParams['full_rot'], resample=Image.BILINEAR)]))
            # Color distortion
            if mdlParams.get('full_color_distort') is not None:
                all_transforms.append(transforms.ColorJitter(brightness=mdlParams.get('brightness_aug', 32. / 255.),
                                                             saturation=mdlParams.get('saturation_aug', 0.5),
                                                             contrast=mdlParams.get('contrast_aug', 0.5),
                                                             hue=mdlParams.get('hue_aug', 0.2)))
            else:
                all_transforms.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
            # Autoaugment
            if self.mdlParams.get('autoaugment', False):
                all_transforms.append(AutoAugment())
            # Cutout
            if self.mdlParams.get('cutout', 0) > 0:
                all_transforms.append(Cutout_v0(n_holes=1, length=self.mdlParams['cutout']))
            # Normalize
            all_transforms.append(transforms.ToTensor())
            all_transforms.append(
                transforms.Normalize(np.float32(self.mdlParams['setMean']), np.float32(self.mdlParams['setStd'])))
            # All transforms
            self.composed = transforms.Compose(all_transforms)
        else:
            self.composed = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(torch.from_numpy(self.setMean).float(),
                                     torch.from_numpy(self.setStd).float())
            ])

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        image = self.names_list[idx]
        name = image['id']
        path = os.path.join(self.image_path[idx], name + '.jpg')
        x = Image.open(path)
        y = image['label']
        meta = self.meta_df.loc[name, self.mdlParams['meta_features']]
        meta_vector = meta.to_numpy()

        # Apply
        x = self.composed(x)
        return (x, meta_vector), y, idx


class Cutout_v0(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        # print(img.shape)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        img = img * np.expand_dims(mask, axis=2)
        img = Image.fromarray(img)
        return img


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        # print("before gather",logpt)
        # print("target",target)
        logpt = logpt.gather(1, target)
        # print("after gather",logpt)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            # print("alpha",self.alpha)
            # print("gathered",at)
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
