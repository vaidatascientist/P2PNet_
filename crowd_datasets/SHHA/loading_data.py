import torchvision.transforms as standard_transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .SHHA import SHHA
import util.misc as utils

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        # standard_transforms.Normalize(mean=[0.7351, 0.6163, 0.5233],
        #                             std=[0.2114, 0.2171, 0.2306]),
    ])
    
    # create the training dataset
    train_set = SHHA(data_root, train=True,
                     transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = SHHA(data_root, train=False, transform=transform)

    return train_set, val_set


class FIBY_Lightning(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, num_workers, pin_memory):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_set, self.val_set = loading_data(self.data_root)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=utils.collate_fn_crowd)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=utils.collate_fn_crowd)

    # def test_dataloader(self):
    #     return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
    #                       num_workers=self.num_workers, pin_memory=self.pin_memory)
