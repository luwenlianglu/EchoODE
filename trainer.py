import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os

from dataloader.cityscapes import CityScapes
#from dataloader.echocardiac import echoCardiac
from util.general_functions import get_model, get_optimizer, make_data_loader, plot_grad_flow, calc_width, count_parameters
from util.lr_scheduler import LR_Scheduler, create_lr_scheduler
from util.losses import get_loss_function, get_reconstruction_loss_function, FocalLoss2d, MultiClassDiceLoss
from util.class_weighting import get_class_weights
from util.evaluator import Evaluator
from util.summary import TensorboardSummary
from core.gan import *
import time

class Trainer(object):

    def __init__(self, args):

        self.args = args
        self.tr_global_step = 1
        self.val_global_step = 1
        self.best_mIoU = -1
        self.bestSegLoss=10000000
        self.bestLvIoU=-1
        self.bestSegLosssave,self.bestLvIoUsave, self.bestsave=None,None,None
        self.num_classes = args.num_classes
        self.mode = args.mode
        self.segmentation = args.segmentation
        self.reconstruct = args.reconstruct

        self.model = get_model(args, args.num_classes)
        # print(self.model)
        self.best_model = copy.deepcopy(self.model)
        self.optimizer = get_optimizer(self.model, args)
        self.summary = TensorboardSummary(args)
        self.log_file = os.path.join(self.summary.output_dir, "log.txt")

        if not args.trainval:
            self.train_loader, self.val_loader = make_data_loader(args, 'train'), make_data_loader(args, 'val')
        else:
            self.train_loader, self.val_loader = make_data_loader(args, 'trainval'), make_data_loader(args, 'test')

        self.class_weights = None#get_class_weights(self.train_loader, self.num_classes, args.weighting_mode) if args.loss_type=="focal" else None

        self.criterion = get_loss_function(args.loss_type, self.class_weights)
        self.dice_cri=MultiClassDiceLoss()
        if self.reconstruct:
            self.reconstruction_criterion = get_reconstruction_loss_function(args.reconstruct_loss_type)
        # self.scheduler = LR_Scheduler(args.lr_policy, args.lr, args.epochs, len(self.train_loader))
        self.scheduler = create_lr_scheduler(self.optimizer, len(self.train_loader), self.args.epochs, warmup=True)
        self.evaluator = Evaluator(self.num_classes)
        self.device=args.device
        self.recInner = self.args.recInner
        self.indices = torch.tensor([i * self.args.time_dilation for i in range(self.args.timesteps)]).to(self.device)
        self.withGAN=self.args.withGAN
        if self.withGAN:
            self.netD_img, self.netD_seq, self.optimizer_netD = create_netD(self.args)
            self.netD_scheduler = create_lr_scheduler(self.optimizer_netD, len(self.train_loader), self.args.epochs, warmup=True)

        with open(self.log_file, 'a') as f:
            try:
                # Attempt to load checkpoint
                checkpoint = torch.load(os.path.join(self.summary.output_dir, "saved_models", "checkpoint.pt"))
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.best_mIoU = checkpoint["best_mIoU"]

                if self.withGAN:
                    self.netD_img.load_state_dict(checkpoint['netD_img'])
                    self.netD_seq.load_state_dict(checkpoint['netD_seq'])
                    self.optimizer_netD.load_state_dict(checkpoint['optimizer_netD'])
                    self.netD_scheduler.load_state_dict(checkpoint['scheduler_netD'])
                self.bestsave=torch.load(os.path.join(self.summary.output_dir, "saved_models", "best.pt"))
                f.write("Resuming run from epoch {}\n".format(self.args.start_epoch))
                print("Resuming run from epoch {}\n".format(self.args.start_epoch))
            except FileNotFoundError:
                for arg in vars(self.args):
                    f.write("{},{}\n".format(arg, getattr(args, arg)))
                f.write("Starting run from scratch\n\n")
                print("Starting run from scratch\n\n")
                if self.args.load_reconstruct:
                    di = args.model + "-" + args.sequence_model + "-" + args.hasTempBottle + (
                        ("-" + "reconstruct") if args.reconstruct else "") + "-" + args.loss_type + "-timesteps" + str(
                        args.timesteps) + "-dilation" + str(args.time_dilation)
                    constructdirectory = os.path.join(r"I:\lwl\lvh-seg\Videoseg_encdec_gruode_forreconstruction\outputs",
                                                      args.dataset, di, "saved_models", "best.pt")
                    constructmodel = torch.load(constructdirectory)
                    self.model.load_state_dict(constructmodel['model'])


    def training(self, epoch):
        train_loss = 0.0
        if self.reconstruct:
            train_reconstruction_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            with torch.autograd.set_detect_anomaly(True):
                imageAll = sample[0].to(self.device)
                image = torch.index_select(imageAll, 1, self.indices)
                target = sample[1].to(self.device)
                maskrank = sample[-1]
                self.optimizer.zero_grad()
                if self.recInner:
                    segmentation_output, reconstruction_output = self.model(image,pred_dur=True)
                    segmentation_output = torch.index_select(segmentation_output, 1, self.indices)
                    reconstruction_target = imageAll
                else:
                    segmentation_output, reconstruction_output = self.model(image)
                    reconstruction_target = image

                if self.segmentation:
                    idx = [[i for i in range(segmentation_output.size(0))], maskrank.tolist()]
                    segmentation_output_mask = segmentation_output[idx]
                    ce_loss = self.criterion(segmentation_output_mask, target.long())
                    diceloss=self.dice_cri(segmentation_output_mask, target)
                    segmentation_loss=ce_loss+diceloss
                    total_loss = segmentation_loss

                if self.reconstruct:
                    reconstruction_loss = self.reconstruction_criterion(reconstruction_output, reconstruction_target)
                    train_reconstruction_loss += reconstruction_loss.item()
                    reconstruction_loss = self.args.reconstruct_loss_coeff * reconstruction_loss
                    total_loss = reconstruction_loss
                    if self.withGAN:
                        loss_netD = self.args.lamb_adv * self.netD_seq.netD_adv_loss(image, reconstruction_output,self.device)
                        loss_netD += self.args.lamb_adv * self.netD_img.netD_adv_loss(image, reconstruction_output,self.device)


                if self.segmentation and self.reconstruct:
                    total_loss = segmentation_loss + reconstruction_loss
                if (i+1)%50==0:
                    with open(self.log_file, 'a') as f:
                        lossinf=f"advloss: {loss_netD.item()/self.args.lamb_adv if self.withGAN else 0} " \
                         f"reconstrctionloss: {(reconstruction_loss.item()/self.args.reconstruct_loss_coeff if self.reconstruct else 0):.4f}  " \
                         f"segloss: {segmentation_loss.item():.4f}" \
                         f"diceloss: {diceloss.item():.4f}\n"
                        f.write(lossinf)



                # Train D
                if self.withGAN:
                    self.optimizer_netD.zero_grad()
                    loss_netD.backward()
                    self.optimizer_netD.step()
                    self.netD_scheduler.step()
                # Train G
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                netD_lr=self.optimizer_netD.param_groups[0]["lr"] if self.withGAN else 0

                if self.args.clip > 0:
                    # if self.args.gpu_ids:
                    #     torch.nn.utils.clip_grad_norm_(self.model.module().parameters(), self.args.clip)
                    # else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)


                train_loss += total_loss.item()
                if self.withGAN:
                    train_loss+=loss_netD.item()
                tbar.set_description("Epoch:{}    Train loss: {:.3f}".format(epoch, train_loss / (i + 1)))


        self.summary.add_scalar('train/total_loss_epoch', train_loss, epoch)
        if self.reconstruct:
            self.summary.add_scalar('train/total_recon_loss_epoch', train_reconstruction_loss, epoch)
            self.summary.add_scalar('train/total_seg_loss_epoch', train_loss - train_reconstruction_loss, epoch)
        # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {(train_loss/len(self.train_loader)):.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"netD_lr: {netD_lr:.6f}\n"
        with open(self.log_file, 'a') as f:
            f.write(train_info)

    def validation(self, epoch):
        test_loss = 0.0
        test_reconstruction_loss = 0.0
        self.model.eval()
        vbar = tqdm(self.val_loader)
        num_img_val = len(self.val_loader)

        labels = []
        outputs = []

        for i, sample in enumerate(vbar):  # inner loop within one epoch
            imageAll = sample[0].to(self.device)
            image = torch.index_select(imageAll, 1, self.indices)
            target = sample[1].to(self.device)
            maskrank = sample[-1]
            with torch.no_grad():
                if self.recInner:
                    segmentation_output, reconstruction_output = self.model(image, pred_dur=True)
                    segmentation_output = torch.index_select(segmentation_output, 1, self.indices)
                    reconstruction_target = imageAll
                else:
                    segmentation_output, reconstruction_output = self.model(image)
                    reconstruction_target = image

            if self.segmentation:
                idx = [[i for i in range(segmentation_output.size(0))], maskrank.tolist()]
                segmentation_output_mask = segmentation_output[idx]
                ce_loss = self.criterion(segmentation_output_mask, target.long())
                diceloss = self.dice_cri(segmentation_output_mask, target)
                segmentation_loss = ce_loss + diceloss
                total_loss = segmentation_loss

            if self.reconstruct:
                reconstruction_loss = self.reconstruction_criterion(reconstruction_output, reconstruction_target)
                test_reconstruction_loss += reconstruction_loss.item()
                reconstruction_loss = self.args.reconstruct_loss_coeff * reconstruction_loss
                total_loss = reconstruction_loss

            if self.segmentation and self.reconstruct:
                total_loss = reconstruction_loss + segmentation_loss

            # Show 10 * 3 inference results each epoch
            # if i % (num_img_val // self.args.summaryFreq) == 0:
            #     self.summary.visualize_image(self.val_global_step, image_GT, target, segmentation_output_GT, reconstruction_output_GT, reconstruction_target_GT, split="val")
            #     self.val_global_step += 1

            test_loss += total_loss.item()
            vbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))

            if self.segmentation:
                outputs.append(torch.argmax(segmentation_output_mask, dim=1).cpu().numpy())
                labels.append(target.cpu().numpy())

        if self.segmentation:
            acc, acc_cls, mIoU, IoU_class, fwavacc, mDice, Dice_class = self.evaluator.evaluate(outputs, labels)
            self.summary.add_results(epoch, mIoU, acc, acc_cls, fwavacc, test_loss, mDice, split="val")


        if self.reconstruct:
            self.summary.add_scalar('val/total_recon_loss_epoch', test_reconstruction_loss, epoch)
            self.summary.add_scalar('val/total_seg_loss_epoch', test_loss - test_reconstruction_loss, epoch)

        save_file = {"model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict(),
                     "epoch": epoch,
                     'best_mIoU': self.best_mIoU,
                     "netD_img": self.netD_img.state_dict() if self.withGAN else None,
                     "netD_seq": self.netD_seq.state_dict() if self.withGAN else None,
                     "optimizer_netD": self.optimizer_netD.state_dict() if self.withGAN else None,
                     "scheduler_netD": self.netD_scheduler.state_dict() if self.withGAN else None,
                     }
        if self.segmentation and IoU_class[4] > self.bestLvIoU:
            self.bestLvIoU = IoU_class[4]
            self.bestLvIoUsave = save_file
            torch.save(save_file, os.path.join(self.summary.output_dir, "saved_models", "bestLvIoU.pt"))
        if self.segmentation and test_loss - test_reconstruction_loss<self.bestSegLoss:
            self.bestSegLoss = test_loss - test_reconstruction_loss
            self.bestSegLosssave = save_file
            torch.save(save_file, os.path.join(self.summary.output_dir, "saved_models", "bestSegLoss.pt"))
        if self.segmentation and mIoU > self.best_mIoU:
            self.best_mIoU = mIoU
            save_file["best_mIoU"]=self.best_mIoU
            self.best_model = copy.deepcopy(self.model)
            self.bestsave = save_file
            torch.save(save_file, os.path.join(self.summary.output_dir, "saved_models", "best.pt"))
        torch.save(save_file, os.path.join(self.summary.output_dir, "saved_models", "checkpoint.pt"))
        val_info = ('global correct: {:.2f}\n'
                    'IoU: {}\n'
                    'Dice: {}\n'
                    'mean IoU: {:.2f}  '
                    'mean Dice: {:.2f}  ').format(acc*100,
                        ['{:.2f}'.format(i) for i in (IoU_class * 100).tolist()],
                        ['{:.2f}'.format(i) for i in (Dice_class * 100).tolist()],
                        mIoU * 100,
                        mDice*100)
        with open(self.log_file, 'a') as f:
            f.write(val_info+"\n\n")

        if 0:
            print(20 * "-")
            for i in range(len(IoU_class)):
                print("IoU/Dice for class " + echoCardiac.classes[i] + " is: " + str(IoU_class[i])+"/"+str(Dice_class[i]))
            print(20 * "-")
            print("Accuray: ", acc)
            print("Class accuracy: ", acc_cls)
            print("FwIoU:", fwavacc)
            print("Mean IoU: ", mIoU)
            print("Best IoU: ", self.best_mIoU)

    def  visualization(self, split):
        print(20 * "-")
        print("Started final Visualization")
        print(20 * "-")

        if self.args.epochs !=0:
            checkpoint = torch.load(os.path.join(self.summary.output_dir, "saved_models", "best.pt"))
            self.model.load_state_dict(checkpoint['model'])
            with open(self.log_file, 'a') as f:
                f.write("Prediction of {} ; Best mIoU run from epoch {}\n".format(split, checkpoint["epoch"]))


        visualization_loss = 0.0
        visualization_reconstruction_loss = 0.0

        self.model.eval()
        if split == 'test' or split == 'demoVideo' or split == 'es' or split == 'ed' or split == "abnormal":
            vis_bar = tqdm(make_data_loader(self.args, split))
        else:
            vis_bar = tqdm(self.val_loader)

        labels = []
        outputs = []
        imgs = []
        paths = []

        test_time = []
        num_samples = []
        for i, sample in enumerate(vis_bar):  # inner loop within one epoch
            imageAll = sample[0].to(self.device)
            num_samples.append(imageAll.size(0))
            image = torch.index_select(imageAll, 1, self.indices)
            target = sample[1].to(self.device)
            maskrank = sample[-1]
            with torch.no_grad():
                if self.recInner:
                    segmentation_output_all, reconstruction_output = self.model(image, pred_dur=True)
                    segmentation_output = torch.index_select(segmentation_output_all, 1, self.indices)
                    reconstruction_target = imageAll
                else:
                    t1=time.time()
                    segmentation_output, reconstruction_output = self.model(image)
                    t2=time.time()
                    test_time.append(t2-t1)
                    reconstruction_target = image

            idx = [[i for i in range(segmentation_output.size(0))], maskrank.tolist()]
            segmentation_output_mask = segmentation_output[idx]
            ce_loss = self.criterion(segmentation_output_mask, target.long())
            diceloss = self.dice_cri(segmentation_output_mask, target)
            segmentation_loss = ce_loss + diceloss


            if self.reconstruct:
                reconstruction_loss = self.reconstruction_criterion(reconstruction_output, reconstruction_target)
                visualization_reconstruction_loss += reconstruction_loss.item()
                reconstruction_loss = self.args.reconstruct_loss_coeff * reconstruction_loss
                total_loss = reconstruction_loss + segmentation_loss
            else:
                total_loss = segmentation_loss

            visualization_loss += total_loss.item()
            vis_bar.set_description('Visualization loss: %.3f' % (visualization_loss / (i + 1)))

            outputs.append(torch.argmax(segmentation_output_mask, dim=1).cpu())
            labels.append(target.cpu())
            imgwithmask = image[idx]
            imgs.append(imgwithmask.cpu())
            if split == 'test':
                for kk in range(image.size(0)):
                    if self.recInner:
                        self.summary.save_test_result(imgs=imageAll[kk, ...].cpu(), reconstruct=reconstruction_output[
                            kk, ...].cpu() if self.reconstruct else None,
                                                      segmentation=torch.argmax(segmentation_output_all[kk, ...], dim=1).cpu(),
                                                      path=sample[2][kk])
                    else:
                        self.summary.save_test_result(imgs=image[kk, ...].cpu(), reconstruct=reconstruction_output[
                            kk, ...].cpu() if self.reconstruct else None,
                                                      segmentation=torch.argmax(segmentation_output[kk, ...], dim=1).cpu(),
                                                      path=sample[2][kk])
            if split == 'test' or split == 'demoVideo':
                for kk in range(image.size(0)):
                    paths.append(sample[2][kk])
        all_time = np.sum(np.array(test_time[1:-1]))
        num_samples = np.sum(np.array(num_samples[1:-1]))
        mean_time = all_time/num_samples
        print(num_samples, " samples, mean test_time: ", mean_time, test_time)

        # if split not in ['test', 'demoVideo']:
        acc, acc_cls, mIoU, IoU_class, fwavacc, mDice, Dice_class = self.evaluator.evaluate([output.numpy() for output in outputs], [label.numpy() for label in labels])
        eachIoUDice = self.evaluator.evaluate_each([output.numpy() for output in outputs], [label.numpy() for label in labels])
        np.savetxt(os.path.join(self.summary.output_dir, split+"_detailDice.txt"), eachIoUDice, fmt="%0.6f")
        outputs = torch.cat(outputs,0)
        labels = torch.cat(labels, 0)
        imgs = torch.cat(imgs,0)
        self.summary.save_visualization_images(outputs, labels, imgs, paths)

        # if split not in ['test', 'demoVideo']:
        print(20 * "-")
        for i in range(len(IoU_class)):
            print("IoU/Dice for class " + echoCardiac.classes[i] + " is: " + str(IoU_class[i]) + "/" + str(Dice_class[i]))
        print(20 * "-")
        print("Accuray: ", acc)
        print("Class accuracy: ", acc_cls)
        print("FwIoU:", fwavacc)
        print("Mean IoU: ", mIoU)
        print("Mean Dice:", mDice)
        print("Best IoU: ", self.best_mIoU)
        test_info = ('split:'+split+'--global correct: {:.2f}\n'
                    'IoU: {}\n'
                    'Dice: {}\n'
                    'mean IoU: {:.2f}   '
                     'mean Dice: {:.2f}').format(acc * 100,
                                               ['{:.2f}'.format(i) for i in (IoU_class * 100).tolist()],
                                               ['{:.2f}'.format(i) for i in (Dice_class * 100).tolist()],
                                               mIoU * 100,
                                               mDice*100)
        with open(self.log_file, 'a') as f:
            f.write(test_info + "\n\n")

    def  predDemoVideo(self):
        print(20 * "-")
        print("Started Pred DemoVideo")
        print(20 * "-")

        if self.args.epochs !=0:
            checkpoint = torch.load(os.path.join(self.summary.output_dir, "saved_models", "best.pt"))
            self.model.load_state_dict(checkpoint['model'])
            with open(self.log_file, 'a') as f:
                f.write("Prediction demovideo; Best mIoU run from epoch {}\n".format(checkpoint["epoch"]))

        self.model.eval()
        loader, eachvideoitems = make_data_loader(self.args, "demoVideo")
        vis_bar = tqdm(loader)
        outputs = []
        rec_outputs=[]

        videorank=0

        for i, sample in enumerate(vis_bar):  # inner loop within one epoch
            image = sample[0]
            if self.args.cuda:
                image = image.to(self.device)
            pred_dur=True
            if self.args.sequence_model=='convlstm':
                pred_dur=False

            with torch.no_grad():
                segmentation_output, reconstruction_output = self.model(input=image, pred_dur=pred_dur)


            outputs.append(torch.argmax(segmentation_output[0,...], dim=1).cpu())
            # rec_outputs.append(reconstruction_output[0, ...].cpu())
            paths=sample[2][0]
            if i==(eachvideoitems[videorank]-1):
                output = torch.squeeze(torch.cat(outputs))
                # rec_output = torch.squeeze(torch.cat(rec_outputs))
                self.summary.save_visualization_demovideos(output, paths)
                # self.summary.save_rec_seg_imgs(rec_output, output,paths)
                outputs=[]
                # rec_outputs=[]
                videorank+=1


    def save_network(self):
        self.summary.save_network(self.best_model)
    def save_network_me(self,epoch):
        torch.save(self.bestsave, os.path.join(self.summary.output_dir, "saved_models",  'best_'+str(epoch+1).zfill(3)+'.pt'))
        torch.save(self.bestLvIoUsave,
                   os.path.join(self.summary.output_dir, "saved_models", 'bestLvIoU_' + str(epoch + 1).zfill(3) + '.pt'))
        torch.save(self.bestSegLosssave,
                   os.path.join(self.summary.output_dir, "saved_models", 'bestSegLoss_' + str(epoch + 1).zfill(3) + '.pt'))

    def load_network(self):
        self.best_model = get_model(self.args)
        self.best_model.load_state_dict(torch.load(''))
