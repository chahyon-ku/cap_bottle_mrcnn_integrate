import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="/home/rpm/Lab/cap_bottle/kpam/cap_bottle_mrcnn_integrate/train_tools/pretrained_models/e2e_mask_rcnn_R-50-FPN_1x.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="/home/rpm/Lab/cap_bottle/kpam/cap_bottle_mrcnn_integrate/train_tools/pretrained_models/e2e_mask_rcnn_R-50-FPN_1x_no_last.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="/home/rpm/Lab/cap_bottle/kpam/cap_bottle_mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_mug.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'],
                             ['cls_score.bias', 'cls_score.weight',
                              'bbox_pred.bias', 'bbox_pred.weight',
                              'mask_fcn_logits.weight', 'mask_fcn_logits.bias'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))


if __name__ == '__main__':
    pass
