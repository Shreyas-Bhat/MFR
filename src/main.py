import os
import argparse
# import warnings

import torch
import torchvision

from utils.logging_config import logger
from pipeline import TrainingPipeline, TestingPipeline, EvaluationPipeline


def main(args):
    # load config file from checkpoint, this will include the training information (epoch, optimizer parameters)
    if args.resume is not None:
        logger.info(f"Resuming checkpoint: {args.resume} ...")
        resumed_checkpoint = torch.load(args.resume)
    else:
        resumed_checkpoint = None
    args.resumed_checkpoint = resumed_checkpoint

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ##################
    # Setup pipeline #
    ##################
    if args.mode == 'train':
        pipeline = TrainingPipeline(args)
    elif args.mode == 'test':
        pipeline = TestingPipeline(args)
    elif args.mode == 'eval':
        pipeline = EvaluationPipeline(args)
    else:
        raise NotImplementedError(f'Mode {args.mode} not defined.')

    ################
    # Run pipeline #
    ################

    pipeline.run()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument(
        '-tc', '--template_config', default=None, type=str,
        help=('Template configuraion file. It should contain all default configuration '
              'and will be overwritten by specified config.')
    )
    parser.add_argument(
        '-sc', '--specified_configs', default=None, type=str, nargs='+',
        help=('Specified configuraion files. They serve as experiemnt controls and will '
              'overwrite template configs.')
    )
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'eval'], default='train')
    parser.add_argument('--saved_keys', default=['data_target', 'model_output'], type=str, nargs='+',
                        help='Specify the keys to save at testing mode.')
    parser.add_argument('--ckpts_subdir', type=str, default='ckpts', help='Subdir name for ckpts saving.')
    parser.add_argument('--outputs_subdir', type=str, default='outputs', help='Subdir name for outputs saving.')
    args = parser.parse_args()

    # Set template config to default if not given
    if args.template_config is None:
        args.template_config = f'configs/template_{args.mode}_config.json'

    return args


if __name__ == '__main__':

    # with warnings.catch_warnings():
    #     warnings.simplefilter('error')

    args = parse_args()
    main(args)
