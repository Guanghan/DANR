from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from dataset.det.indoor_centernet import CenterIndoorDataset
from dataset.det.indoor_render256native_centernet import CenterIndoorRender256NativeDataset
from dataset.det.indoor_render512pixelSS_centernet import CenterIndoorRender512PixelSSDataset

default_collate = torch.utils.data._utils.collate.default_collate

import os, sys
import os.path as osp

# Add lib to PYTHONPATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)


def get_dataset(opt, dataset_name="openimgs"):
    if dataset_name == "indoor":
        train_dataset = CenterIndoorDataset(opt, split='train')
        val_dataset = CenterIndoorDataset(opt, split='val')
    elif dataset_name == "indoor_render256native":
        train_dataset = CenterIndoorRender256NativeDataset(opt, split='train')
        val_dataset = CenterIndoorRender256NativeDataset(opt, split='val')
    elif dataset_name == "indoor_render512pixelSS":
        train_dataset = CenterIndoorRender512PixelSSDataset(opt, split='train')
        val_dataset = CenterIndoorRender512PixelSSDataset(opt, split='val')
    else:
        print("Unknown dataset.")
        return NotImplementedError
    return train_dataset, val_dataset


def get_distributed_dataloader(train_dataset, val_dataset):
    if train_dataset is not None:
        print('Setting up training set...')
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas= len(opt.gpus),
            rank= opt.local_rank,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True
        )
    else:
        print('Training set skipped...')
        train_loader = None

    print('Setting up validation set...')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    return train_loader, val_loader

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

def get_distributed_concat_dataloader(train_dataset_list, val_dataset_list):

    if train_dataset_list is not None:
        print('Setting up training set...')
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            torch.utils.data.ConcatDataset(train_dataset_list),
            num_replicas= len(opt.gpus),
            rank= opt.local_rank,

        )
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(train_dataset_list),
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        print('Training set skipped...')
        train_loader = None

    print('Setting up validation set...')
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_dataset_list),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    return train_loader, val_loader, train_sampler


def train(model, optimizer, trainer, train_loader, val_loader, train_sampler, opt, logger):

    best = 1e10
    for epoch in range(opt.cur_epoch + 1, opt.num_epochs + 1):
        train_sampler.set_epoch(epoch)

        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer, opt.local_rank)

            print('Performing validation...')
            if epoch % (2 * opt.val_intervals) == 0:
                with torch.no_grad():
                    log_dict_val, preds = trainer.val_for_eval(epoch, val_loader)
            else:
                with torch.no_grad():
                    log_dict_val, preds = trainer.val(epoch, val_loader)

            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))

            if log_dict_val[opt.metric] < best:
                print('Overwriting the current best model...')
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model, rank=opt.local_rank)
        else:
            print('Finishing...Saving the last model...')
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer, rank=opt.local_rank)

        logger.write('\n')
        if epoch in opt.lr_step:
            print('Saving model...')
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer, rank=opt.local_rank)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return


def validate(trainer, val_loader, opt):
    if opt.test:
        print('Performing validation...')
        _, preds = trainer.val(0, val_loader)
        print('Evaluating accuracy...')
        val_loader.dataset.run_eval(preds, opt.save_dir)
    return


if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    opt = opts().init()
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda:{}'.format(opt.local_rank))    # distributed

    """ 1. network and trainer """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    opt.cur_epoch = 0
    if opt.load_model != '':
        print('Loading model...')
        model, optimizer, opt.cur_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step, opt.local_rank)

    from trainers.det_trainer import CtdetTrainer
    trainer = CtdetTrainer(opt, model, optimizer)

    torch.cuda.set_device(opt.local_rank)
    world_size = len(opt.gpus)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=opt.local_rank,
    )
    torch.manual_seed(42)
    trainer.set_device_distributed(opt.gpus, opt.device)   # distributed


    """ 2. Dataset """
    train_dataset_indoor, val_dataset_indoor = get_dataset(opt, dataset_name="indoor")
    train_dataset_indoor_render, val_dataset_indoor_render = get_dataset(opt, dataset_name="indoor_render512pixelSS")
    train_dataset_list = [train_dataset_indoor, train_dataset_indoor_render]
    val_dataset_list = [val_dataset_indoor]

    if opt.test:
        _, val_loader = get_distributed_dataloader(None, val_dataset_indoor)

        print('Start validating...')
        validate(trainer, val_loader, opt)
    else:
        train_loader, _, train_sampler = get_distributed_concat_dataloader(train_dataset_list, val_dataset_list)
        _, val_loader = get_distributed_dataloader(None, val_dataset_indoor)

        """ 3. Training """
        print('Starting training...')
        train(model, optimizer, trainer, train_loader, val_loader, train_sampler, opt, logger)
        print('Training done.')

    logger.close()


