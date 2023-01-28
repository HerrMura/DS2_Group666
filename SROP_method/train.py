import argparse
import os
import copy
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import math
from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter,tensor2img,calc_psnr
from tqdm import tqdm
from ssim import SSIM
from model import SROP
from offset_map import Offs_Pro
import xlwt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, default='SROP_method/train_results/output')
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('train_x'+str(args.scale),cell_overwrite_ok=True)
    savepath = 'loss_result.xls'

    if args.scale==2:
        train_file='SROP_method/img_datasets/train_datasets_x2.h5'
        eval_file='SROP_method/img_datasets/eval_datasets_x2.h5'
    elif args.scale==3:
        train_file='SROP_method/img_datasets/train_datasets_x3.h5'
        eval_file='SROP_method/img_datasets/eval_datasets_x3.h5'
    elif args.scale==4:
        train_file='SROP_method/img_datasets/train_datasets_x4.h5'
        eval_file='SROP_method/img_datasets/eval_datasets_x4.h5'
        
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.cuda.manual_seed(args.seed)
    
    model=SROP(num_channels=3,scale=args.scale).to(device)
    criterion1 = SSIM(channels=3).to(device)
    criterion2 = torch.nn.MSELoss()
    Mapping_Offset=Offs_Pro().to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    
    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_ssim = 0.0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        epoch_loss1=AverageMeter()
        epoch_loss2=AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            for data in train_dataloader:
                lr_inputs, hr_inputs = data
                    # lr_inputs is of size (batch_size,channels,h,w)
                    # hr_inputs is of size (batch_size,channels,sh,sw)
                lr_inputs = lr_inputs.to(device)
                hr_inputs = hr_inputs.to(device)
                
                offset_map = model(lr_inputs)
                    # offset_map is of size (batch_size,2,sh,sw)              
                sr_train=Mapping_Offset(lr_inputs,offset_map,scale=args.scale,Train=True)
                    # sr_train is of size (batch,channels,sh,sw)
                
                loss1=criterion1(sr_train,hr_inputs,as_loss=True).requires_grad_(True)             
                loss2=criterion2(sr_train,hr_inputs)*0.002
                loss=loss1**2+loss2**2   #使用复合loss
                #loss=criterion1(sr_train,hr_inputs,as_loss=True).requires_grad_(True)
                epoch_loss1.update(loss1,len(hr_inputs))
                epoch_loss2.update(loss2,len(hr_inputs))
                epoch_losses.update(loss, len(hr_inputs))
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(lr_inputs))
            sheet.write(epoch,0,epoch_losses.avg.item())
            sheet.write(epoch,1,epoch_loss1.avg.item())
            sheet.write(epoch,2,epoch_loss2.avg.item())
                
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        
        model.eval()
        with torch.no_grad():
            epoch_ssim = AverageMeter()
            for data in eval_dataloader:
                lr_inputs, hr_inputs = data
                    # lr_inputs is of size (batch_size,channels,h,w)
                    # hr_inputs is of size (batch_size,channels,sh,sw)
                lr_inputs = lr_inputs.to(device)
                hr_inputs = hr_inputs.to(device)
                
                offset_map = model(lr_inputs)
                sr_train=Mapping_Offset(lr_inputs,offset_map,scale=args.scale,Train=False)
                # sr_train is of size (batch_size,channels,sh,sw)
                epoch_ssim.update(calc_psnr(hr_inputs,sr_train), len(sr_train)) # using psnr as the judging value
                #epoch_ssim.update(criterion1(sr_train,hr_inputs,as_loss=False).item(), len(sr_train))
            print('eval ssim: {:.6f}'.format(epoch_ssim.avg))
               
            if epoch_ssim.avg > best_ssim:
                best_epoch = epoch
                best_ssim = epoch_ssim.avg
                best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, ssim: {:.6f}'.format(best_epoch, best_ssim))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    book.save(savepath)
