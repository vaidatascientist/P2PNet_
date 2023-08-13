import warnings
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from engine import *
from models import build_model
from tensorboardX import SummaryWriter
from crowd_datasets.SHHA.loading_data import FIBY_Lightning

warnings.filterwarnings('ignore')


def main(args):
    pl.seed_everything(42, workers=True)
    print(args)

    best_mae_checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        dirpath=args.checkpoints_dir,
        filename='best_rmse_model-{epoch:02d}-{val_rmse:.2f}',
        save_top_k=1,
        mode='min',
        every_n_epochs=1
    )

    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename='latest_model-{epoch:02d}-{val_rmse:.2f}',
    )
    
    model = build_model(args, training=True)
    
    logger = TensorBoardLogger(save_dir='./logs', name='P2PNet')
    dm = FIBY_Lightning(args.data_root, args.batch_size,
                        args.num_workers, args.pin_memory)
    trainer = pl.Trainer(devices=4, accelerator="gpu", logger=logger,
                         strategy="ddp_find_unused_parameters_true",
                         callbacks=[best_mae_checkpoint_callback, latest_checkpoint_callback])
    trainer.fit(model, dm, ckpt_path=args.resume if args.resume else None)

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # # Model parameters
    # parser.add_argument('--frozen_weights', type=str, default=None,
    #                     help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./DATA_ROOT',
                        help='path where the dataset is')

    # parser.add_argument('--output_dir', default='./log',
    #                     help='path where to save, empty for no saving')
    
    parser.add_argument('--checkpoints_dir', default='./weights',
                        help='path where to save checkpoints, empty for no saving')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--pin_memory', default=True,
                        type=bool, help='pin_memory')
    # parser.add_argument('--gpu_id', default=[0], type=list, help='the gpu used for training')

    # parser.add_argument('--transfer_weights_path', default=None, type=str)

    # parser.add_argument('--enable_checkpoint', default=None, type=bool)
    
    # args = parser.parse_args([])
    # import pickle

    # with open("./weights/args_training.pkl", "wb") as f:
    #     pickle.dump(args, f)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
