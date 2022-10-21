from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle as pkl
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

import _init_paths
from config import config
from config import update_config
from core.function import test_ED
from core.loss import build_criterion
from dataset.build import build_ED_dataloader
from dataset import RealLabelsImagenet
from dataset.build import _build_ED_dataset
from models import build_model
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import strip_prefix_if_present

import matplotlib.image as mpimg

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test_ED classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'test_ED')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))

    model = build_model(config)
    model.to(torch.device('cuda'))

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        raise ValueError("Unknown model file")

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    summary_model_on_master(model, config, final_output_dir, False)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    '''
    # define loss function (criterion) and optimizer
    criterion = build_criterion(config, train=False)
    criterion.cuda()
    
    valid_loader = build_dataloader(config, False, args.distributed)
    '''
    real_labels = None

    valid_labels = None

    logging.info('=> start testing ED')
    start = time.time()
    # Read class2idx txt
    txt = open(config.ED.CLASS2INDEX_PATH, 'r').read()
    tmp = txt[1:-1].split(',')
    para = {}
    for i in tmp:
        chara, index = i.split(':')[0][2], int(i.split(':')[1])
        #print(chara, index)
        para[index] = chara
    # print(para)
    # make new txt
    str_result = open('OUTPUT/mthv2/cvt-13-224x224/str_reslut.txt', 'a')
    # test_ED
    subroots = os.listdir(config.ED.ROOT)
    logging.info("sentences number:"+str(len(subroots)))
    for subroot in subroots[69398:]:
        logging.info("=> " + subroot)
        file_root = os.path.join(config.ED.ROOT, subroot) + '/'
        print(file_root)
        if(file_root == []):
            continue

        ED_dataset = _build_ED_dataset(config, False, file_root)
        print("len ED", len(ED_dataset))
        valid_loader = build_ED_dataloader(config, False, args.distributed, ED_dataset)
        #print('valid_loader:',len(valid_loader))
        pred_string = ''
        for i, x in enumerate(valid_loader):
            # compute output
            x = x['img'].cuda(non_blocking=True)

            outputs = model(x)
            if(len(outputs.shape)>1):
                pred = torch.max(outputs, dim = 1)[1].cpu().numpy()
                pred_str = ''.join([para[i] for i in pred])
            else:
                # one character
                pred = torch.max(outputs, dim = 0)[1].cpu().numpy()
                print(pred, type(pred))
                pred_str = str(para[int(pred)]) 
            # free
            logging.info("=> " + pred_str)
            x.cpu() 
            pred_string += pred_str          
            
        str_result.writelines(subroot + ' ' + pred_string + '\n') 

    '''
        for i in img_names:
            print(os.path.join(file_root, i[0]))
            img = torch.from_numpy(mpimg.imread(os.path.join(file_root, i[0])))
            output = model(img.cuda())

            chara = torch.max(output, dim = 1)[1].cpu().numpy()
            print(output)

        
  
    test_ED(config, valid_loader, model, criterion,
         final_output_dir, tb_log_dir, writer_dict,
         args.distributed, real_labels=real_labels,
         valid_labels=valid_labels)
    '''
    logging.info('=> test duration time: {:.2f}s'.format(time.time()-start))

    writer_dict['writer'].close()
    logging.info('=> finish testing')
    logging.info('=> this is test_ED')


if __name__ == '__main__':
    main()
