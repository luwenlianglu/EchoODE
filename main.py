from trainer import Trainer
from parser_options import ParserOptions
from util.general_functions import print_training_info
import torch
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)


def main(model_name="echo_ode"):
    args = ParserOptions().parse()  # get training options

    if model_name == "echo_lstm":
        args.with_skip = 1
        args.onlysegskip = 1
        args.skip_from = "rec"
        args.sequence_model = "convlstm"
        args.reconstruct = 1

    if model_name == "convlstm":
        args.with_skip = 1
        args.onlysegskip = 1
        args.skip_from = "enc"
        args.sequence_model = "convlstm"
        args.reconstruct = 0

    trainer = Trainer(args)

    # print_training_info(args)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)

        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    if args.segmentation:
        trainer.visualization(args.vis_split)
        trainer.predDemoVideo()
        trainer.save_network()

    trainer.summary.writer.add_scalar('val/bestmIoU', trainer.best_mIoU, args.epochs)
    trainer.summary.writer.close()

if __name__ == "__main__":
    #model_name choice ["echo_ode", "echo_lstm", "convlstm"]
   main(model_name="echo_ode")
