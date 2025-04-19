from utils import log
from utils import util
from model import DAFCN
from utils.opt import Options
from utils import h36motion3d as datasets

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def main(opt):
    """
    Main function to initialize the model, prepare datasets, run training and evaluation.
    """
    lr_now = opt.lr_now
    start_epoch = 1

    print('>>> Creating model...')
    net_pred = DAFCN(in_features=opt.in_features,
                     kernel_size=opt.kernel_size,
                     d_model=opt.d_model,
                     num_stage=opt.num_stage,
                     dct_n=opt.dct_n).cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=lr_now)
    print(">>> Total parameters: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1e6))

    # Load checkpoint if needed
    if opt.is_load or opt.is_eval:
        ckpt_path = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> Loading checkpoint from '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> Checkpoint loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> Loading datasets...')
    # Dataset splits
    if not opt.is_eval:
        train_dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset size: {:d}'.format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=16, pin_memory=True)

        val_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset size: {:d}'.format(len(val_dataset)))
        val_loader = DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=16, pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset size: {:d}'.format(len(test_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Evaluation only
    if opt.is_eval:
        result_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        log.save_csv_log(opt, np.array(list(result_test.keys())), np.array(list(result_test.values())), is_create=True, file_name='test_walking')
        return

    # Training loop
    best_err = 1000.0
    for epoch in range(start_epoch, opt.epoch + 1):
        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
        print(f"\n>>> Training epoch {epoch}...")

        train_metrics = run_model(net_pred, optimizer, is_train=0, data_loader=train_loader, epo=epoch, opt=opt)
        print(f"Train MPJPE: {train_metrics['m_p3d_h36']:.3f}")

        val_metrics = run_model(net_pred, is_train=1, data_loader=val_loader, epo=epoch, opt=opt)
        print(f"Validation MPJPE: {val_metrics['m_p3d_h36']:.3f}")

        test_metrics = run_model(net_pred, is_train=3, data_loader=test_loader, epo=epoch, opt=opt)
        print(f"Test Error (#1): {test_metrics['#1']:.3f}")

        # Log saving
        log_keys, log_values = ['epoch', 'lr'], [epoch, lr_now]
        for metrics, prefix in zip([train_metrics, val_metrics, test_metrics], ['', 'valid_', 'test_']):
            for k, v in metrics.items():
                log_keys.append(prefix + k)
                log_values.append(v)
        log.save_csv_log(opt, np.array(log_keys), np.array(log_values), is_create=(epoch == 1))

        # Save best checkpoint
        is_best = val_metrics['m_p3d_h36'] < best_err
        if is_best:
            best_err = val_metrics['m_p3d_h36']
        log.save_ckpt({
            'epoch': epoch,
            'lr': lr_now,
            'err': val_metrics['m_p3d_h36'],
            'state_dict': net_pred.state_dict(),
            'optimizer': optimizer.state_dict()
        }, is_best, opt)

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    """
    Run model for one epoch. Supports training, validation, and testing.
    """
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    total_loss = 0
    total_mpjpe = 0 if is_train <= 1 else np.zeros(opt.output_n)
    n_samples = 0

    dim_used = opt.dim_used  # predefined useful joint indices
    index_to_ignore = opt.index_to_ignore
    index_to_equal = opt.index_to_equal

    iters = 1
    kernel_input = opt.kernel_size
    total_seq = kernel_input + opt.output_n

    for i, (batch_3d) in enumerate(data_loader):
        batch_size, seq_len, _ = batch_3d.shape
        if batch_size == 1 and is_train == 0:
            continue

        batch_3d = batch_3d.cuda().float()
        input_sup = batch_3d[:, :, dim_used][:, -total_seq:].reshape(batch_size, total_seq, -1, 3)
        input_src = batch_3d[:, :, dim_used]

        # model forward
        pred_output = net_pred(input_src, input_n=opt.input_n, output_n=opt.output_n, itera=iters)

        gt_output = batch_3d[:, opt.input_n:opt.input_n + opt.output_n]
        gt_output[:, :, dim_used] = pred_output[:, kernel_input:, 0]
        gt_output[:, :, index_to_ignore] = gt_output[:, :, index_to_equal]
        gt_output = gt_output.reshape(batch_size, opt.output_n, 32, 3)
        batch_3d = batch_3d.reshape(batch_size, opt.input_n + opt.output_n, 32, 3)

        # Training loss
        if is_train == 0:
            loss = torch.mean(torch.norm(pred_output[:, :, 0] - input_sup, dim=3))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net_pred.parameters(), max_norm=opt.max_norm)
            optimizer.step()
            total_loss += loss.item() * batch_size

        # Evaluation metric
        if is_train <= 1:
            mpjpe = torch.mean(torch.norm(batch_3d[:, opt.input_n:] - gt_output, dim=3))
            total_mpjpe += mpjpe.item() * batch_size
        else:
            mpjpe = torch.mean(torch.norm(batch_3d[:, opt.input_n:] - gt_output, dim=3), dim=2)
            total_mpjpe += torch.sum(mpjpe, dim=0).cpu().numpy()

    result = {}
    if is_train == 0:
        result['l_p3d'] = total_loss / n_samples
    if is_train <= 1:
        result['m_p3d_h36'] = total_mpjpe / n_samples
    else:
        mean_per_joint_errors = total_mpjpe / n_samples
        for j in range(opt.output_n):
            result[f"#{j+1}"] = mean_per_joint_errors[j]

    return result


if __name__ == '__main__':
    options = Options().parse()
    main(options)
