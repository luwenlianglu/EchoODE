import torch
import numpy as np

class Evaluator(object):
    def __init__(self, num_classes, fast_eval=True):
        self.num_classes = num_classes
        self.fast_eval = fast_eval

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        hist = np.zeros((self.num_classes, self.num_classes))
        for lp, lt in zip(predictions, gts):
            hist += self._fast_hist(lp.flatten(), lt.flatten())

        # axis 0: gt, axis 1: prediction
        acc = np.diag(hist).sum() / hist.sum()

        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu[1:])
        dice = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        mean_dice = np.nanmean(dice[1:])

        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, iu, fwavacc, mean_dice, dice

    def evaluate_each(self, predictions, gts):
        eachIoUDice=[]
        for lp, lt in zip(predictions, gts):
            for b in range(lp.shape[0]):
                hist = self._fast_hist(lp[b].flatten(), lt[b].flatten())
                dia=np.diag(hist)
                sum1=hist.sum(axis=1)
                sum2=hist.sum(axis=0)
                iu1=dia[4]/(sum1[4]+sum2[4]-dia[4]) if  (sum1[4]+sum2[4]-dia[4]) else np.nan
                dice1=2*dia[4]/(sum1[4]+sum2[4]) if  (sum1[4]+sum2[4]) else np.nan
                eachIoUDice.append([iu1, dice1])
        return np.array(eachIoUDice)