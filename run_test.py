import argparse
import sys
import pickle

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from engine import *
from models.p2pnet import P2PNet
import os
import warnings
warnings.filterwarnings('ignore')

from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    
    # load_path = "./args_training.pkl"
    # with open(load_path, "rb") as f:
    #     loaded_args = pickle.load(f)
    
    model = build_model(args, training=False)
    
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    del checkpoint['state_dict']['criterion.empty_weight']
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])

    if debug:
        sys.exit()

    model.eval()
    model.cuda(args.gpu_id)
    
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    # image_folder = "/home/ubuntu/P2PNet/DATA_ROOT/test/images"
    # image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(".jpg")]
    
    # predict_cnt = 0
    # total_dev = 0
    
    # for img_path in image_paths:
    img_path = "./DATA_ROOT/test/frame_00100.jpg"
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())
    
    # if predict_cnt != 81:
    #     total_dev += abs(predict_cnt - 81)

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 3
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        # save the visualized image
        cv2.imwrite(os.path.join(args.output_dir, 'pred_lightning{}.jpg'.format(predict_cnt)), img_to_draw)
    # print(total_dev/len(image_paths))
    # print(predict_cnt)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)