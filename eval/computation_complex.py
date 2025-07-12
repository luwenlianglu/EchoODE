import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os

from dataloader.cityscapes import CityScapes
from dataloader.echocardiac import echoCardiac
from util.general_functions import get_model, get_optimizer, make_data_loader, plot_grad_flow, calc_width, count_parameters
from util.lr_scheduler import LR_Scheduler, create_lr_scheduler
from util.losses import get_loss_function, get_reconstruction_loss_function, FocalLoss2d, MultiClassDiceLoss
from util.class_weighting import get_class_weights
from util.evaluator import Evaluator
from util.summary import TensorboardSummary
from core.gan import *
from parser_options import ParserOptions
from core.convlstm_new import ConvLSTM
import time


def resnet_test():
    import torchvision.models as models
    model = models.resnet18()
    a = torch.randn(5, 3, 224, 224)
    # input = (3,224,224)
    b = model(a)

    """通过torchstat.stat 可以查看网络模型的参数量和计算复杂度FLOPs"""
    from torchstat import stat
    # stat(model, (3, 224, 224))

    """thop工具包仅支持FLOPs和参数量的计算"""
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)  # 46388784.0 561706.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 46.389M 561.706K

    """ptflops统计 参数量 和 FLOPs"""
    from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
    #                                          verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # """torchsummary 用来计算网络的计算参数等信息"""
    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3,224,224))

def echoode_test():
    args = ParserOptions().parse()
    args.gpu_ids="0"
    device = args.device
    model = get_model(args, args.num_classes)
    model = model.to(device)

    a = torch.randn(1, 4, 3, 192, 256).to(device)
    t1 = time.time()
    b = model(a)
    t2 = time.time()


    """通过torchstat.stat 可以查看网络模型的参数量和计算复杂度FLOPs"""
    from torchstat import stat
    # stat(model, (4, 3, 192, 256))

    """thop工具包仅支持FLOPs和参数量的计算"""
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 4, 3, 192, 256).to(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)  # 46388784.0 561706.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 46.389M 561.706K\
    print(t2-t1)

    """ptflops统计 参数量 和 FLOPs"""
    from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
    #                                          verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # """torchsummary 用来计算网络的计算参数等信息"""
    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3,224,224))

def convlstm_test_old():
    model = ConvLSTM(3, 7, (3, 3), 5, True, True, False)

    a = torch.rand((1, 4, 3, 192, 256))
    b = model(a)

    """通过torchstat.stat 可以查看网络模型的参数量和计算复杂度FLOPs"""
    from torchstat import stat
    # stat(model, (4, 3, 192, 256))

    """thop工具包仅支持FLOPs和参数量的计算"""
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 4, 3, 192, 256)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)  # 46388784.0 561706.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 46.389M 561.706K

def convlstm_test():
    args = ParserOptions().parse()
    args.with_skip=1
    args.onlysegskip=1
    args.skip_from = "enc"
    args.sequence_model = "convlstm"
    args.reconstruct=0
    args.gpu_ids="0"

    device = args.device
    model = get_model(args, args.num_classes)
    model = model.to(device)

    a = torch.randn(1, 4, 3, 192, 256).to(device)
    t1 = time.time()
    b = model(a)
    t2 = time.time()

    """通过torchstat.stat 可以查看网络模型的参数量和计算复杂度FLOPs"""
    from torchstat import stat
    # stat(model, (4, 3, 192, 256))

    """thop工具包仅支持FLOPs和参数量的计算"""
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 4, 3, 192, 256).to(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)  # 46388784.0 561706.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 46.389M 561.706K
    print(t2-t1)

    """ptflops统计 参数量 和 FLOPs"""
    from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
    #                                          verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # """torchsummary 用来计算网络的计算参数等信息"""
    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3,224,224))

def echolstm_test():
    args = ParserOptions().parse()

    args.with_skip = 1
    args.onlysegskip = 1
    args.skip_from = "rec"
    args.sequence_model = "convlstm"
    args.reconstruct = 1
    args.gpu_ids="0"

    device = args.device
    model = get_model(args, args.num_classes)
    model = model.to(device)

    a = torch.randn(1, 4, 3, 192, 256).to(device)
    t1 = time.time()
    b = model(a)
    t2 = time.time()

    """通过torchstat.stat 可以查看网络模型的参数量和计算复杂度FLOPs"""
    from torchstat import stat
    # stat(model, (4, 3, 192, 256))

    """thop工具包仅支持FLOPs和参数量的计算"""
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 4, 3, 192, 256).to(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)  # 46388784.0 561706.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 46.389M 561.706K
    print(t2-t1)

    """ptflops统计 参数量 和 FLOPs"""
    from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
    #                                          verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # """torchsummary 用来计算网络的计算参数等信息"""
    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3,224,224))


if __name__ == "__main__":
    # resnet_test()
    echoode_test()
    convlstm_test()
    echolstm_test()