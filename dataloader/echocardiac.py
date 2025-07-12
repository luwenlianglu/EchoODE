import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils import data
import util.custom_transforms as custom_transforms
import torchvision.transforms as standard_transforms
import json

root = "D:\\lwl\\lvh-seg\\data\\videos"

visualize = standard_transforms.ToTensor()
cwdpath=os.getcwd()
palette_path = os.path.join(cwdpath, "palette.json")
# palette_path =  "./palette.json"
# print(cwdpath)
# print(palette_path)
assert os.path.exists(palette_path), f"palette {palette_path} not found."
with open(palette_path, "rb") as f:
    pallette_dict = json.load(f)
    palette = []
    for v in pallette_dict.values():
        palette += v

CITYSCAPES_CLASS_COLOR_MAPPING = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
    -1: (0, 0, 142),
    255: (0, 0, 0),
}

TRAINID_TO_ID = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17,
                 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24,
                 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 255: 255}

MY_CITYSCAPES_CLASS_COLOR_MAPPING = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    255: 255
}

visualize = standard_transforms.ToTensor()

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def colorize_mask_submit(mask):
    # mask: numpy array of the mask
    new_mask = np.random.rand(256,512)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            new_mask[i][j] = TRAINID_TO_ID[int(mask[i][j])]

    return Image.fromarray(new_mask.astype(np.uint8), 'L')

def colorize_mask_color(mask):
    # mask: numpy array of the mask
    new_mask = np.random.rand(512,256,3)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            new_mask[i][j] = CITYSCAPES_CLASS_COLOR_MAPPING[int(mask[i][j])]

    return Image.fromarray(new_mask.astype(np.uint8))

class echoCardiac_segnotlast(data.Dataset):

    num_classes = 7
    classes = ["BackGround", "LA", "V-V", "LV-P", "LV", "RV", "RV-I"]

    def __init__(self,seqLen, quality, split, blank=False, input_size=512, base_size=1024, time_dilation=1, reconstruct=0, shuffle=0, sequence_model="convgru_ode"):
        self.rootDir = "I:\\lwl\\lvh-seg\\data\\videos"
        self.maskDir = "I:\\lwl\\lvh-seg\\data\\allDataWithGenerate\\masks"
        self.seqLen = seqLen
        self.ignore_label = 255
        self.quality = quality
        self.split = split
        self.reconstruct = reconstruct
        self.shuffle = shuffle
        self.blank = blank
        self.input_size = input_size
        self.base_size = base_size
        self.time_dilation = time_dilation
        self.sequence_model=sequence_model

        if split == 'demoVideo':
            self.imgs = self.make_demo_dataset()
        else:
            self.imgs = self.make_dataset()

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform, self.input_transform, self.target_transform = self.get_transforms()
    def get_transforms(self):
        mean_std = ([0.2079, 0.2079, 0.2079], [0.2081, 0.2081, 0.2081])
        if self.split == 'train':
            joint = custom_transforms.Compose([
                # custom_transforms.Resize(self.input_size),
                custom_transforms.randomResize(self.input_size[1]),
                custom_transforms.randomCrop(self.input_size),
                custom_transforms.RandomHorizontallyFlip(),
                custom_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                custom_transforms.RandomGaussianBlur()
            ])
        elif self.split == 'val' or self.split == 'test' or self.split == 'ed' or self.split == 'es' or self.split == 'abnormal' or self.split == 'demoVideo':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size),
            ])
        else:
            raise RuntimeError('Invalid dataset mode')

        '''
        if self.split == 'train':
            joint = custom_transforms.Compose([
                custom_transforms.Scale(self.base_size),
                custom_transforms.RandomCrop(size=self.input_size),
                custom_transforms.RandomHorizontallyFlip(),
                custom_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                custom_transforms.RandomGaussianBlur(),
            ])
        elif self.split == 'val':
            joint = custom_transforms.Compose([
                custom_transforms.Scale(self.base_size)
            ])
        else:
            raise RuntimeError('Invalid dataset mode')
        '''

        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        target_transform = custom_transforms.MaskToTensor()

        return joint, input_transform, target_transform

    def __getitem__(self, index):
       return self.get_item_sequence(index)

    def get_item_sequence(self, index):
        if self.split=="demoVideo":
            img_paths, mask_path = self.imgs[index]
            images = []

            if self.shuffle:
                copy = img_paths[:len(img_paths) - 1]
                random.shuffle(copy)
                img_paths[:len(img_paths) - 1] = copy

            for img_path in img_paths:
                images.append(Image.open(img_path).convert('RGB'))

            mask = Image.open(mask_path)

            if self.joint_transform is not None:
                images, mask = self.joint_transform(images, mask)

            if self.input_transform is not None:
                for i in range(len(images)):
                    images[i] = self.input_transform(images[i])

            if self.blank:
                images[len(images) - 1][True] = 0

            if self.target_transform is not None:
                mask = self.target_transform(mask)

            # if self.reconstruct:
            #     reconstruct_images = [image.clone() for image in images]
            # images.pop()
            # reconstruct_images.pop(0)
            # reconstruct_images = torch.stack(reconstruct_images)

            if self.quality == 'fbf-1234':
                images = torch.cat(images, dim=0)
            else:
                images = torch.stack(images)

            if self.reconstruct:
                return images, mask, img_path.split('\\')[-2] + "_" + img_path.split('\\')[-1]
            else:
                return images, mask, img_path.split('\\')[-2] + "_" + img_path.split('\\')[-1]
        else:
            vidname, vidlen, maskrank = self.imgs[index]
            f=(maskrank-1)//self.time_dilation+1
            g=self.seqLen-((vidlen-maskrank)//self.time_dilation)
            # 这里k取[1-8]，后面返回的时候index-1会有利于后续代码书写
            seqrank=random.randint(max(1,g), min(self.seqLen,f))
            if self.split=="test" or self.split=="val" or self.split=="es" or self.split=="ed":
                seqrank=min(self.seqLen, f)
            # indices = [self.time_dilation*i for i in range(1-seqrank,self.seqLen+1-seqrank)]
            indices = [self.time_dilation * (1 - seqrank) + i for i in range(self.time_dilation * (self.seqLen - 1) + 1)]
            img_paths = []
            for i in indices:
                img_paths.append(os.path.join(self.rootDir, "videoWithSplitImages", vidname,
                                              str(maskrank + i).zfill(3) + ".jpg"))
            mask_path = os.path.join(self.maskDir, vidname + "_" + str(maskrank).zfill(3) + ".png")
            if self.shuffle:
                copy = img_paths[:len(img_paths) - 1]
                random.shuffle(copy)
                img_paths[:len(img_paths) - 1] = copy
            images = []
            for img_path in img_paths:
                images.append(Image.open(img_path).convert('RGB'))

            mask = Image.open(mask_path)
            if self.joint_transform is not None:
                images, mask = self.joint_transform(images, mask)

            if self.input_transform is not None:
                for i in range(len(images)):
                    images[i] = self.input_transform(images[i])

            if self.blank:
                images[len(images) - 1][True] = 0

            if self.target_transform is not None:
                mask = self.target_transform(mask)

            # if self.reconstruct:
            #     reconstruct_images = [image.clone() for image in images]
            # images.pop()
            # reconstruct_images.pop(0)
            # reconstruct_images = torch.stack(reconstruct_images)

            if self.quality == 'fbf-1234':
                images = torch.cat(images, dim=0)
            else:
                images = torch.stack(images)

            # return images, mask, img_path.split('\\')[-2] + "_" + img_path.split('\\')[-1],seqrank-1
            return images, mask, img_path.split('\\')[-2] + "_" + str(maskrank).zfill(3)+".jpg", seqrank - 1

    def __len__(self):
        return len(self.imgs)

    def make_dataset(self):
        with open(os.path.join(self.rootDir, "split", self.split+".txt")) as f:
            all = f.readlines()
            all = [x.strip() for x in all if len(x.strip()) > 0]
        items = []
        count = 0
        for video in all:
            splist = video.split("|")
            for i in range(len(splist)-3):
                items.append((splist[0], int(splist[-1]), int(splist[i+1])))
        return items

    def make_demo_dataset(self):
        with open(os.path.join(self.rootDir, "split", self.split+".txt")) as f:
            all = f.readlines()
            all = [x.strip() for x in all if len(x.strip()) > 0]
        if self.sequence_model == "convlstm":
            self.eachvideoitems = []
            summm = 0
            eachlen = self.time_dilation * self.seqLen
            if 'sequence' in self.quality or self.quality == 'fbf-1234':
                indices = [i * self.time_dilation for i in range(self.seqLen)]
                finalitems = []
                for i in range(len(all)):
                    splist = all[i].split("|")
                    name = splist[0]
                    video_len = int(splist[-1])
                    summm += video_len // eachlen * self.time_dilation
                    self.eachvideoitems.append(summm)
                    for j in range(video_len // eachlen):
                        for k in range(self.time_dilation):
                            images = []
                            for i in indices:
                                images.append(os.path.join(self.rootDir, "videoWithSplitImages", name,
                                                           str(eachlen * j + 1 + i+k).zfill(3) + ".jpg"))
                            finalitems.append((images, os.path.join(self.rootDir, "videoWithSplitImages", name,
                                                                    str(1).zfill(3) + ".jpg")))
        else:
            self.eachvideoitems=[]
            summm=0
            eachlen = self.time_dilation*(self.seqLen-1)+1
            if 'sequence' in self.quality or self.quality == 'fbf-1234':
                indices = [i * self.time_dilation for i in range(self.seqLen)]
                finalitems = []
                for i in range(len(all)):
                    splist = all[i].split("|")
                    name = splist[0]
                    video_len = int(splist[-1])
                    summm+=video_len//eachlen
                    self.eachvideoitems.append(summm)
                    for j in range(video_len//eachlen):
                        images = []
                        for i in indices:
                            images.append(os.path.join(self.rootDir, "videoWithSplitImages", name, str(eachlen*j+1+i).zfill(3)+".jpg"))
                        finalitems.append((images, os.path.join(self.rootDir, "videoWithSplitImages", name, str(1).zfill(3)+".jpg")))

        return finalitems

if __name__ == "__main__":
    b=0
