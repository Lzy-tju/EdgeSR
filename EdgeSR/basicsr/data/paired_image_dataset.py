import random

import cv2
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torchvision import transforms
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms import InterpolationMode
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = 0
        self.std = 1


        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        # if self.file_client is None:
        #     self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index%len(self.paths)]['gt_path']
        gt_image = cv2.imread(gt_path,0)
        # img_bytes = self.file_client.get(gt_path, 'gt')
        # img_gt = imfrombytes(img_bytes, flag="grayscale",float32=True)
        # lq_path = self.paths[index]['lq_path']
        # lq_image = cv2.imread(lq_path, 0)
        # img_bytes = self.file_client.get(lq_path, 'lq')
        # img_lq = imfrombytes(img_bytes, flag="grayscale",float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            x0 = random.randint(0, gt_image.shape[-2] - gt_size)
            y0 = random.randint(0, gt_image.shape[-1] - gt_size)
            crop_hr = gt_image[x0: x0 + gt_size, y0: y0 + gt_size]
            crop_lr = resize_fn(crop_hr, gt_size//scale)

        if self.opt['use_aug']:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(transforms.ToTensor()(crop_hr))
            # crop_hr = augment(crop_hr)  # Apply augmentation to tensor
        elif self.opt['phase'] == 'val' or "test":
            # If it's the validation phase, use the whole image (no random crop)
            hr_h=(gt_image.shape[0]//scale)*scale
            hr_w=(gt_image.shape[1]//scale)*scale
            lr_h = (gt_image.shape[0] // scale)
            lr_w = (gt_image.shape[1] // scale)
            crop_hr = gt_image[ : hr_h, : hr_w]
            crop_lr = resize_fn(crop_hr, (lr_h,lr_w))
            # crop_hr = gt_image
            # crop_lr = resize_fn(crop_hr, crop_hr.shape[0] // scale)

            # Convert crop_hr and crop_lr to tensor for validation phase
            if isinstance(crop_hr, np.ndarray):  # Ensure it's a numpy array before converting to tensor
                crop_hr = transforms.ToTensor()(crop_hr)  # Convert to tensor
            if isinstance(crop_lr, np.ndarray):  # Ensure it's a numpy array before converting to tensor
                crop_lr = transforms.ToTensor()(crop_lr)  # Convert to tensor

            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_aug'])

        # color space transform
        # if 'color' in self.opt and self.opt['color'] == 'y':
        #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
        #     img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(crop_lr, self.mean, self.std, inplace=True)
            normalize(crop_hr, self.mean, self.std, inplace=True)

        return {'lq': crop_lr, 'gt': crop_hr, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)*20 if self.opt['phase'] == 'train' else len(self.paths)
