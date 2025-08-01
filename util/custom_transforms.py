import types
import random
import torch

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import functional as F
from torchvision import transforms as T
import numpy as np

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, mask):
        img = images[0]
        assert img.size == mask.size

        for t in self.transforms:
            images, mask = t(images, mask)
        return images, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, images, mask):
        if self.padding > 0:
            images = [ImageOps.expand(image, border=self.padding, fill=0) for image in images]
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        img = images[0]

        assert img.size == mask.size
        w, h = img.size
        tw, th = self.size

        if w == tw and h == th:
            return images, mask
        if w < tw or h < th:
            return [image.resize((tw, th), Image.BILINEAR) for image in images], mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        cropped_images, cropped_masks =  [image.crop((x1, y1, x1 + tw, y1 + th)) for image in images], mask.crop((x1, y1, x1 + tw, y1 + th))
        return cropped_images, cropped_masks

class RandomHorizontallyFlip(object):
    def __call__(self, images, mask):
        if random.random() < 0.5:
            return [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images], mask.transpose(Image.FLIP_LEFT_RIGHT)
        return images, mask

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, mask):
        img = images[0]

        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return images, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return [image.resize((ow, oh), Image.BILINEAR) for image in images], mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return [image.resize((ow, oh), Image.BILINEAR) for image in images], mask.resize((ow, oh), Image.NEAREST)

class randomResize(object):
    def __init__(self, min_size):
        self.min_size = min_size+20
        self.max_size = self.min_size+100

    def __call__(self, images, mask):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        images = [F.resize(image, size) for image in images]
        mask = F.resize(mask, size, T.InterpolationMode.NEAREST)
        return images, mask

class randomCrop(object):
    def __init__(self, size):
        self.size = (size[1], size[0])

    def __call__(self, images, mask):
        crop_params = T.RandomCrop.get_params(images[0], self.size)
        images = [F.crop(image, *crop_params) for image in images]
        mask = F.crop(mask, *crop_params)
        return images, mask


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, mask):
        img = images[0]

        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size[0]) or (h >= w and h == self.size[1]):
            return images, mask

        return [image.resize(self.size, Image.BILINEAR) for image in images], mask.resize(self.size, Image.NEAREST)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0, multiscale=False):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.multiscale=multiscale

    def __call__(self, images, mask):
        img = images[0]

        # random scale (short edge)
        if self.multiscale:
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            w, h = img.size
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)

            images = [image.resize((ow, oh), Image.BILINEAR) for image in images]
            mask = mask.resize((ow, oh), Image.NEAREST)

            # pad crop
            if short_size < self.crop_size:
                padh = self.crop_size - oh if oh < self.crop_size else 0
                padw = self.crop_size - ow if ow < self.crop_size else 0
                images = [ImageOps.expand(image, border=(0, 0, padw, padh), fill=0) for image in images]
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_sizet
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        images = [image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size / 2)) for image in images] #TODO RADU
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return images, mask

class RandomGaussianBlur(object):
    def __call__(self, images, mask):
        if random.random() < 0.5:
            radius = random.random()
            images = [image.filter(ImageFilter.GaussianBlur(radius=radius)) for image in images]
        return images, mask

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        return transforms

    @staticmethod
    def forward_transforms(image, transforms):
        for transform in transforms:
            image = transform(image)

        return image

    def __call__(self, images, mask):
        """
        Args:
            images (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transforms = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return [self.forward_transforms(img, transforms) for img in images], mask