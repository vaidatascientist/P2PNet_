import random
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageEnhance


class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "16x_v5_train.list"
        self.eval_list = "16x_v5_val.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        
        # apply augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        
        # if self.train:
            # New Augmentations
            # if random.random() > 0.5:
            #     img, point = random_rotation(img, point)
        #     if random.random() > 0.5:
        #         img = adjust_brightness(img)
        #     if random.random() > 0.5:
        #         img = random_blur(img)
            # if random.random() > 0.5:
            #     img = adjust_contrast_tensor(img)
            # if random.random() > 0.5:
            #     img = inject_noise_tensor(img)
        #     if random.random() > 0.5:
        #         img = perspective_transform_tensor(img)
            # if random.random() > 0.5:
            #     img = color_jitter(img)
                
        # random crop augumentaiton
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
                
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]
                
        # if random.random() > 0.5 and self.train and self.flip:
        #     # random flip vertically
        #     img = torch.Tensor(img.flip(2))
        #     for i, _ in enumerate(point):
        #         point[i][:, 1] = 128 - point[i][:, 1]
        
        # if random.random() > 0.5 and self.train and self.flip:
        #     # horizontal flip
        #     if random.random() > 0.5:
        #         img = torch.Tensor(img[:, :, :, ::-1].copy())
        #         for i, _ in enumerate(point):
        #             point[i][:, 0] = 128 - point[i][:, 0]
                
        #         if random.random() > 0.5:
        #             img = torch.Tensor(img.flip(2))
        #             for i, _ in enumerate(point):
        #                 point[i][:, 1] = 128 - point[i][:, 1]
                    
        #     # vertical flip
        #     else:
        #         img = torch.Tensor(img[:, :, ::-1, :].copy())
        #         for i, _ in enumerate(point):
        #             point[i][:, 1] = 128 - point[i][:, 1]
                
        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    return img, np.array(points)

def random_crop(img, den, num_patch=3):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den

def random_rotation(img_tensor, points, range_degree=(-10, 10)):
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    angle = random.uniform(range_degree[0], range_degree[1])
    rows, cols, _ = img_np.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img_np = cv2.warpAffine(img_np, M, (cols, rows))
    rotated_img_tensor = torch.tensor(rotated_img_np.transpose(2, 0, 1))
    rotated_points = []
    for (x, y) in points:
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        rotated_points.append((new_x, new_y))
    return rotated_img_tensor, rotated_points

def adjust_brightness(img_tensor, factor_range=(0.75, 1.25)):
    factor = random.uniform(factor_range[0], factor_range[1])
    brightened_img_tensor = img_tensor * factor
    brightened_img_tensor = torch.clamp(brightened_img_tensor, 0, 1)
    return brightened_img_tensor

def adjust_contrast_tensor(img_tensor, factor_range=(0.75, 1.25)):
    img_pil = transforms.ToPILImage()(img_tensor)
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(img_pil)
    contrasted_img = enhancer.enhance(factor)
    contrasted_tensor = transforms.ToTensor()(contrasted_img)
    return contrasted_tensor

def color_jitter(img, hue=0.1, saturation=0.1):
    jitter = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=saturation, hue=hue)
    img = jitter(img)
    return img

def inject_noise_tensor(img_tensor, noise_factor=0.05):
    noise = torch.randn_like(img_tensor) * noise_factor

    # Add noise to the image
    noisy_img_tensor = img_tensor + noise

    # Clip values to be in the range [0, 1]
    noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)

    return noisy_img_tensor

def perspective_transform_tensor(img_tensor, max_warp=0.1):
    # Convert tensor to numpy for perspective transform
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    h, w, _ = img_np.shape

    # Randomly define four points for source and destination perspective
    src_pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)

    warp_shift = np.random.uniform(-max_warp, max_warp, size=src_pts.shape) * np.array([w, h])
    dst_pts = src_pts + warp_shift

    # Ensure points are in the correct data type
    dst_pts = dst_pts.astype(np.float32)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transform
    warped_img_np = cv2.warpPerspective(img_np, M, (w, h))

    # Convert the numpy image back to tensor
    warped_img_tensor = torch.tensor(warped_img_np.transpose(2, 0, 1))

    return warped_img_tensor

def random_blur(img_tensor, max_ksize=3):
    """
    Applies a Gaussian blur with a random kernel size to the given tensor image.
    """
    # Randomly select a kernel size (must be odd)
    ksize = random.choice([i for i in range(1, max_ksize+1) if i % 2 == 1])
    
    # Create Gaussian kernel
    x_coord = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    x_grid = x_coord.repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = torch.tensor([0., 0.])
    variance = torch.tensor([1.5, 1.5])
    gaussian_kernel = (1 / (2. * np.pi * variance[0] * variance[1])) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / 
                          (2 * variance[0] * variance[1])
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to 2D depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, ksize, ksize)
    gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
    
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    blurred_img_tensor = F.conv2d(img_tensor, gaussian_kernel, stride=1, padding=ksize//2, groups=3)
    blurred_img_tensor = blurred_img_tensor.squeeze(0)  # Remove batch dimension
    
    return blurred_img_tensor