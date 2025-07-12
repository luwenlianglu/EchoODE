import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import glob
from util.general_functions import tensor2im, tensor2submit_image
import numpy as np
from PIL import Image
import cv2
from dataloader import echocardiac

class TensorboardSummary(object):
    def __init__(self, args):
        self.args = args
        self.output_dir = self.generate_directory(args)
        self.experiment_dir = os.path.join(self.output_dir, "events")
        self.save_model_dir = os.path.join(self.output_dir, "saved_models")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.save_model_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.experiment_dir))

    def generate_directory(self, args):
        checkname = 'debug_' if args.debug else ''
        checkname += args.model

        if 'deeplab' in args.model:
            checkname += '_pretrained' if args.pretrained_resnet else ''
            checkname += '_os:' + str(args.output_stride)

        if 'unet' in args.model:
            checkname += '_num-downs-' + str(args.num_downs) + '_ngf-' + str(args.ngf) + '_down-type-' + str(args.down_type)

        checkname += '_mode-' + args.mode

        if 'sequence' in args.mode:
            checkname += '_seq-model-' + args.sequence_model

            if args.sequence_stacked_models > 1:
                checkname += '_stacked-' + str(args.sequence_stacked_models)

            if args.sequence_model == 'lstm':
                if args.lstm_bidirectional:
                    checkname += '_bidirectional-True'

                if args.lstm_initial_state != '0':
                    checkname += '_init-state-' + args.lstm_initial_state

            if 'tcn' in args.sequence_model:
                checkname += '_num-levels-tcn-' + str(args.num_levels_tcn)
                checkname += '_tcn-kernel-size-' + str(args.tcn_kernel_size)

        checkname += '_init-type-' + args.init_type
        checkname += '_optim-' + args.optim + '_lr-' + str(args.lr)

        if args.clip > 0:
            checkname += '_clipping-' + str(args.clip)

        if args.reconstruct:
            checkname += '_reconstruct-' + args.reconstruct_loss_type + '-' + str(args.reconstruct_loss_coeff)
            checkname += '_rec_remove-skip-1' if args.reconstruct_remove_skip else ''

        checkname += '_blanked' if args.blank else ''
        checkname += '_seg-removed_skip-1' if args.remove_skip else ''
        checkname += '_resize-' + '-'.join([str(x) for x in list(args.resize)])

        if args.loss_type == 'focal':
            checkname += '_focal'

        if args.use_class_weights:
            checkname += '_weighted'

        if 'sequence' in args.mode:
            checkname += '_td-' + str(args.time_dilation)

        checkname += '_epochs-' + str(args.epochs)
        checkname += '_trainval-1' if args.trainval else ''

        checkname = ""

        # directory = os.path.join(args.results_dir, args.dataset, args.model+"-"+args.sequence_model, checkname)
        tt = args.temporal_layer_list
        temlayer = ""
        for i in range(len(tt)):
            temlayer = temlayer + str(tt[i])
        # di = args.sequence_model+('-'+str(args.resize[0])+'_'+str(args.resize[1])) + ("-" + "withGAN"+'_'+str(args.lamb_adv) if args.withGAN else "") + \
        #      (("-" + "reconstruct"+'_'+str(args.reconstruct_loss_coeff)) if args.reconstruct else "")+"-" + args.loss_type+"-timesteps"+str(args.timesteps)+\
        #      "-dilation"+str(args.time_dilation)+("-load_reconstruct" if args.load_reconstruct else "")+('-BS'+str(args.batch_size))+\
        #      ('-skip' if args.with_skip else "")+('-attention' if args.with_attention else "")+('-reduceLayer' if args.reduce_downsample else "")
        # directory = os.path.join(args.results_dir,args.dataset+"_"+args.optim+str(args.lr),di,checkname)#args.optim+str(args.lr)
        di = args.sequence_model + (("-" + "reconstruct"+'_'+str(args.reconstruct_loss_coeff)) if args.reconstruct else "")+"-" +\
             "-timesteps"+str(args.timesteps)+ "-dilation"+str(args.time_dilation)+"-epoch"+str(args.epochs)+('-skipfrom_'+args.skip_from if args.with_skip else "")+\
             ('_onlyseg' if (args.onlysegskip and args.with_skip) else "")+('_recInner' if args.recInner else "")+('-deLayer' if args.reduce_downsample else "")+\
             ('-DUA' if args.with_attention else "") +('-TUA' if args.with_seq_attention else "")
        directory = os.path.join(args.results_dir,di)#args.optim+str(args.lr)
        runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
        # experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))
        experiment_dir = directory

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        return experiment_dir

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_results(self, epoch, mIOU, acc, acc_class, fwavacc, test_loss, mDice, split="train"):
        self.writer.add_scalar(split + '/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar(split + '/mIoU', mIOU, epoch)
        self.writer.add_scalar(split + '/Acc', acc, epoch)
        self.writer.add_scalar(split + '/Acc_class', acc_class, epoch)
        self.writer.add_scalar(split + '/Forward_acc', fwavacc, epoch)
        self.writer.add_scalar(split + '/mDice', mDice, epoch)

    def visualize_image(self, epoch, image, target, output=None, reconstructed_image=None, reconstruction_target=None, split="train"):
        if output is not None:
            output = torch.argmax(output, dim=1)
            output[(target == 255)] = 255

        images = []
        outputs = []
        targets = []
        reconstructed_images = []
        reconstruction_targets = []

        number_of_images = min(5, image.size(0))

        for i in range(number_of_images):
            if output is not None:
                outputs.append(tensor2im(output[i]))
                targets.append(tensor2im(target[i]))

            if not self.args.blank:
                images.append(tensor2im(image[i]))
            else:
                images.append(image[i])

            if reconstructed_image is not None:
                reconstructed_images.append(tensor2im(reconstructed_image[i]))
                reconstruction_targets.append(tensor2im(reconstruction_target[i]))

        grid_image = make_grid(images)
        self.writer.add_image(split + '/ZZ Image', grid_image, epoch)

        if output is not None:
            grid_image = make_grid(outputs)
            self.writer.add_image(split + '/Predicted label', grid_image, epoch)

            grid_image = make_grid(targets)
            self.writer.add_image(split + '/Groundtruth label', grid_image, epoch)

        if reconstructed_image is not None:
            grid_image = make_grid(reconstructed_images)
            self.writer.add_image(split + '/ZZ Recon Image', grid_image, epoch)

            grid_image = make_grid(reconstruction_targets)
            self.writer.add_image(split + '/ZZ Recon Target', grid_image, epoch)

    def save_test_result(self, imgs, reconstruct,segmentation, path=''):
        outputs_save_dir = os.path.join(self.output_dir, "img-rec-seg")

        if not os.path.isdir(outputs_save_dir):
            os.makedirs(outputs_save_dir)

        mean,std = [0.2079, 0.2079, 0.2079], [0.2081, 0.2081, 0.2081]

        # if len(paths) == 0:
        # outputs[(targets == 255)] = 255
        src = []
        recons=[]
        segs = []
        for i in range(imgs.size(0)):
            img = imgs[i].numpy().transpose((1, 2, 0))
            img = img * np.array(std) + np.array(mean)
            img = (img * 255).astype('uint8')
            seg = tensor2im(segmentation[i], return_tensor=False)
            seg = seg.astype('uint8')
            src.append(img)
            src.append(255 * np.ones((img.shape[0], 5, 3)))
            segs.append(seg)
            segs.append(255 * np.ones((img.shape[0], 5, 3)))

            if reconstruct is not None:
                recon = reconstruct[i].numpy().transpose((1, 2, 0))
                recon = recon * np.array(std) + np.array(mean)
                recon = (recon * 255).astype('uint8')
                recons.append(recon)
                recons.append(255 * np.ones((img.shape[0], 5, 3)))

        src=np.concatenate(src, axis=1)
        segs = np.concatenate(segs, axis=1)
        if reconstruct is not None:
            recons = np.concatenate(recons, axis=1)
            all = np.concatenate([src,255 * np.ones((5,(img.shape[1]+5)*imgs.size(0), 3)),recons, 255 * np.ones((5,(img.shape[1]+5)*imgs.size(0), 3)), segs], axis=0)
        else:
            all = np.concatenate([src, 255 * np.ones((5, (img.shape[1] + 5) * imgs.size(0), 3)), segs], axis=0)

        name = path

        # all.save(outputs_save_dir + '/' + name)
        cv2.imencode('.png', all)[1].tofile(outputs_save_dir + '/' + name)

    def save_visualization_images(self, outputs, targets, imgs, paths=''):
        outputs_save_dir = os.path.join(self.output_dir, "prediction")

        if not os.path.isdir(outputs_save_dir):
            os.makedirs(outputs_save_dir)

        # targets_save_dir = os.path.join(self.output_dir, 'targets')
        # save_targets = False
        #
        # if not os.path.isdir(targets_save_dir):
        #     save_targets = True
        #     os.makedirs(targets_save_dir)

        mean,std = [0.2079, 0.2079, 0.2079], [0.2081, 0.2081, 0.2081]

        # if len(paths) == 0:
        # outputs[(targets == 255)] = 255

        for i in range(outputs.size(0)):
            if self.args.submit_format:
                output = tensor2submit_image(outputs[i])
            else:
                output = tensor2im(outputs[i], return_tensor=False)

            if not self.args.submit_format:
                output = output.astype('uint8')
            target = tensor2im(targets[i], return_tensor=False)
            target = target.astype('uint8')
            img = imgs[i].numpy().transpose((1, 2, 0))
            img = img * np.array(std) + np.array(mean)
            img = (img * 255).astype('uint8')
            all = np.concatenate(
                [img, 255 * np.ones((img.shape[0], 5, 3)), target, 255 * np.ones((img.shape[0], 5, 3)), output], axis=1)

            if len(paths) == 0:
                name = str(i) + '.png'
            else:
                name = paths[i]

            # all.save(outputs_save_dir + '/' + name)
            cv2.imencode('.png', all)[1].tofile(outputs_save_dir + '/' + name)

    def save_visualization_demovideos(self, outputs, paths=''):
        outputs_save_dir = os.path.join(self.output_dir, "demoVideo",'videos')
        name = paths.split("_")[0]
        if not os.path.isdir(outputs_save_dir):
            os.makedirs(outputs_save_dir)
        imgspath = os.path.join(self.output_dir, "demoVideo", 'images', name)
        if not os.path.isdir(imgspath):
            os.makedirs(imgspath)
        imgwidth = outputs.size(2)
        imgheight = outputs.size(1)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(os.path.join(outputs_save_dir, name + ".avi"), fourcc, 30.0, (imgwidth, imgheight))

        if self.args.sequence_model == "convlstm":
            timesteps = self.args.timesteps
            time_dilation = self.args.time_dilation
            eachlen = timesteps * time_dilation
            rank = 0
            for k in range(outputs.size(0) // eachlen):
                for j in range(timesteps):
                    for i in range(time_dilation):
                        output = tensor2im(outputs[eachlen * k + timesteps * i + j], return_tensor=False).astype(
                            'uint8')
                        video.write(output)
                        out1 = outputs[eachlen * k + timesteps * i + j].numpy()
                        out1 = echocardiac.colorize_mask(out1)
                        out1.save(imgspath + '/' + str(rank).zfill(3) + '.png')
                        rank += 1
        else:
            for i in range(outputs.size(0)):
                output = tensor2im(outputs[i], return_tensor=False).astype('uint8')
                video.write(output)
                out1=outputs[i].numpy()
                out1 = echocardiac.colorize_mask(out1)
                out1.save(imgspath + '/' + str(i).zfill(3) + '.png')
        video.release()
    def save_rec_seg_imgs(self, rec_outputs, outputs, paths=''):

        name = paths.split("_")[0]
        outputs_save_dir = os.path.join(self.output_dir, "demoVideo", name)
        if not os.path.isdir(outputs_save_dir):
            os.makedirs(outputs_save_dir)
        imgwidth = outputs.size(2)
        imgheight = outputs.size(1)

        mean, std = [0.2079, 0.2079, 0.2079], [0.2081, 0.2081, 0.2081]
        if outputs.size(0)!=rec_outputs.size(0):
            print("seg and rec size is different, return.")
            return

        for i in range(outputs.size(0)):
            recon = rec_outputs[i].numpy().transpose((1, 2, 0))
            recon = recon * np.array(std) + np.array(mean)
            recon = (recon * 255).astype('uint8')

            seg = tensor2im(outputs[i], return_tensor=False)
            seg = seg.astype('uint8')

            unite = np.concatenate([recon,255 * np.ones((5,imgwidth, 3)),seg], axis=0)
            name_pic = name+'_'+str(i+1).zfill(3)+".jpg"

            # all.save(outputs_save_dir + '/' + name)
            cv2.imencode('.png', unite)[1].tofile(outputs_save_dir + '/' + name_pic)

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # video = cv2.VideoWriter(os.path.join(outputs_save_dir, name + ".avi"), fourcc, 30.0, (imgwidth, imgheight))
        #
        # for i in range(outputs.size(0)):
        #     output = tensor2im(outputs[i], return_tensor=False).astype('uint8')
        #     video.write(output)
        # video.release()

    def save_network(self, model):
        # path = self.args.save_dir + '/' + self.experiment_dir.replace('./', '')
        path = os.path.join(self.output_dir, "saved_models")
        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(model.state_dict(), path + '/' + 'network.pth')
