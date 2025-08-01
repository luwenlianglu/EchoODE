import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import functools
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dataloader import cityscapes
from dataloader import echocardiac
#from dataloader import echo_dynamic
from core.deeplabv3_plus import DeepLabv3_plus
from core.pspnet import PSPNet
from core.unet import UNet
from core.unet_ode import UNet_ODE
from core.unet_paper import UNet_paper
from core.unet_pytorch import UNet_torch
from core.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from core.sync_batchnorm.replicate import patch_replication_callback
from core.convlstm_new import ConvLSTM


def make_data_loader(args, split='train'):
    if args.dataset == "echocardiac":
        if split == 'train':
            set = echocardiac.echoCardiac_segnotlast(args.timesteps, args.mode, 'train', args.blank, args.resize, args.base_size, args.time_dilation, args.reconstruct, args.shuffle)
            loader = DataLoader(set, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        elif split == 'val':
            set = echocardiac.echoCardiac_segnotlast(args.timesteps, args.mode, 'val', args.blank, args.resize, args.base_size, args.time_dilation, args.reconstruct, args.shuffle)
            loader = DataLoader(set, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        elif split == 'test' or split == 'ed' or split == 'es' or split == 'abnormal':
            set = echocardiac.echoCardiac_segnotlast(args.timesteps, args.mode, split, args.blank, args.resize, args.base_size, args.time_dilation, args.reconstruct, args.shuffle)
            loader = DataLoader(set, batch_size=16, num_workers=8, shuffle=False, pin_memory=True)
        elif split == 'trainval':#args.batch_size
            train_set = echocardiac.echoCardiac_segnotlast(args.timesteps, args.mode, 'train', args.blank, args.resize, args.base_size, args.time_dilation, args.reconstruct, args.shuffle)
            val_set = echocardiac.echoCardiac_segnotlast(args.timesteps, args.mode, 'val', args.blank, args.resize, args.base_size, args.time_dilation, args.reconstruct, args.shuffle)
            trainval_set = ConcatDataset([train_set, val_set])
            loader = DataLoader(trainval_set, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        elif split == 'demoVideo':
            set = echocardiac.echoCardiac_segnotlast(args.timesteps, args.mode, split, args.blank, args.resize, args.base_size, args.time_dilation, args.reconstruct, args.shuffle, args.sequence_model)
            loader = DataLoader(set, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
            eachvideoitems = set.eachvideoitems
            return loader, eachvideoitems
    else:
        print("there is no this datasetname! please caution")
    return loader


def get_model(args, num_classes=19):
    """
    Builds the model based on the provided arguments and returns the initialized model

        Parameters:
        args (argparse)    -- command line arguments
        num_classes (int)  -- number of possible classes
    """

    norm_layer = get_norm_layer(args.norm_layer)
    model = UNet_ODE(num_classes=num_classes, args=args, norm_layer=norm_layer)


    if len(args.gpu_ids)>1:
            model = torch.nn.DataParallel(model, device_ids=[int(x) for x in args.gpu_ids])
            patch_replication_callback(model)

    if args.cuda:
        model = model.to(args.device)

    return model

def get_optimizer(model, args):
    """
    Builds the optimizer for the model based on the provided arguments and returns the optimizer

        Parameters:
        model          -- the network to be optimized
        args           -- command line arguments
    """
    if len(args.gpu_ids)>1:
        train_params = model.module.get_train_parameters(args.lr)
    else:
        train_params = model.get_train_parameters(args.lr)

    if args.optim == 'sgd':
        optimizer = optim.SGD(train_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    elif args.optim == 'adam':
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    return optimizer


def get_norm_layer(norm_type='instance'):
    """Returns a normalization layer

        Parameters:
            norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'syncbn':
        norm_layer = SynchronizedBatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_model(net, device, init_type='normal', init_gain=0.02):
    """Initialize the network weights

    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """

    net = net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, '_all_weights') and (classname.find('LSTM') != -1 or classname.find('GRU') != -1):
            for names in m._all_weights:
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(m, name)
                    nn.init.xavier_normal_(weight.data, gain=init_gain)

                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    nn.init.constant_(bias.data, 0.0)

                    if classname.find('LSTM') != -1:
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        nn.init.constant_(bias.data[start:end], 1.)
        elif classname.find('BatchNorm2d') != -1 or classname.find('SynchronizedBatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('Initialized network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def tensor2submit_image(input_image):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
    image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    image = cityscapes.colorize_mask_submit(image_numpy)
    return image

def tensor2im(input_image, imtype=np.uint8, return_tensor=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.ndim == 2:
            image_numpy = np.array(echocardiac.colorize_mask(image_numpy).convert("RGB")).transpose((2,0,1))/255.0
        elif image_numpy.ndim == 3:
            image_numpy = (image_numpy - np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = image_numpy* 255.0
        # image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    return torch.from_numpy(image_numpy.astype(imtype)) if return_tensor else np.transpose(image_numpy, (1,2,0))

def plot_grad_flow(model):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n and "norm" not in n):
            name = n[:n.find('.network')] + n[n.find('.conv'):n.find('.weight')] if 'network' in n else n[:n.find('.weight')]
            layers.append(name)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")

    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.01)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", alpha=0.5, lw=4),
                Line2D([0], [0], color="b", alpha=0.5, lw=4)], ['max-gradient', 'mean-gradient'])
    plt.tight_layout()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_width(net):
    net_params = filter(lambda p: p[1].requires_grad, net.named_parameters())
    weight_count = 0
    for (name, param) in net_params:
        weight_count += np.prod(param.size())
    return weight_count

def print_training_info(args):
    print('Segmentation', args.segmentation)

    if args.shuffle:
        print('Shuffling', args.shuffle)

    if 'unet' in args.model:
        print('Ngf', args.ngf)
        print('Num downs', args.num_downs)
        print('Down type', args.down_type)

        if args.remove_skip:
            print('Remove skip connections', args.remove_skip)

    print('Mode', args.mode)

    if 'sequence' in args.mode:
        print('Sequence model', args.sequence_model)
        print('Number stacked sequence models', args.sequence_stacked_models)

        if args.sequence_model == 'lstm':
            print('LSTM Bidirectional', args.lstm_bidirectional)
            print('LSTM initial state', args.lstm_initial_state)

        if 'tcn' in args.sequence_model:
            print('TCN num levels', args.num_levels_tcn)
            print('TCN kernel level size', args.tcn_kernel_size)

        print('Time dilation', args.time_dilation)

    print('Optimizer', args.optim)
    print('Learning rate', args.lr)

    if args.clip > 0:
        print('Gradient clip', args.clip)

    if args.reconstruct:
        print('Reconstruct', args.reconstruct)
        print('Reconstruct coeff', args.reconstruct_loss_coeff)
        print('Reconstruct loss function', args.reconstruct_loss_type)
        print('Reconstruct remove skip connections', args.reconstruct_remove_skip)

    print('Resize', args.resize)
    print('Blank', args.blank)
    print('Batch size', args.batch_size)
    print('Norm layer', args.norm_layer)
    print('Using cuda', torch.cuda.is_available())

    if args.use_class_weights:
        print('Using weighted ' + args.loss_type + ' loss')
    else:
        print('Using ' + args.loss_type + ' loss')

    if args.loss_type == 'focal':
        print('Gamma', args.gamma)
        # print('Alpha', args.alpha)

    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)

