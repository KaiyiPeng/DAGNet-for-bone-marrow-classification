from cell_loader import Cell_PathLoader, CellLoader
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from networks2 import densenet201
from networks2_2 import DAGNet
from networks3_1 import resNeXt101_64x4d, resNeXt50_32x4d

def train_func(model_name='DAGNet', resume=False, base_lr=0.001, batch_size=32, reset_lr_epoch=5,
               folder_num=5, folder=0, cell_list_path='./archive/cell_list'):
    #################

    # network and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('model name : ' + model_name + '     resume : ' + str(resume)
          + '   folder / folder_num : ' + str(folder) + '/' + str(folder_num)
          + '     device : ' + device + '   batch size : ' + str(batch_size))
    if model_name == 'densenet':
        MBC_Networks = densenet201(pretrained=resume, num_classes=21)
    if model_name == 'DAGNet':
        MBC_Networks = DAGNet(pretrained=resume, num_classes=21)
    if model_name == 'resNeXt101':
        MBC_Networks = resNeXt101_64x4d(pretrained=resume, num_classes=21)
    if model_name == 'resNeXt50':
        MBC_Networks = resNeXt50_32x4d(pretrained=resume, num_classes=21)

    MBC_Networks = MBC_Networks.to(device)
    if device == 'cuda':
        MBC_Networks = torch.nn.DataParallel(MBC_Networks)
        cudnn.benchmark = False
    optimizer = optim.SGD(MBC_Networks.parameters(), lr=base_lr, momentum=0.9)
    # loss
    CEloss = nn.CrossEntropyLoss()
    # train and validate data loader
    CPL = Cell_PathLoader(cell_list_path=cell_list_path)
    train_cell_data, val_cell_data, weight_list = CPL.Load_Cell_list(folder_num=folder_num, folder=folder)
    train_data_sampler = torch.utils.data.sampler.RandomSampler(train_cell_data)
    CL_tra = CellLoader(cell_path_list=train_cell_data)
    train_data = torch.utils.data.DataLoader(CL_tra, batch_size=batch_size, shuffle=False,
                                             sampler=train_data_sampler, num_workers=5)
    CL_val = CellLoader(cell_path_list=val_cell_data)
    val_data_sampler = torch.utils.data.sampler.SequentialSampler(val_cell_data)
    val_data = torch.utils.data.DataLoader(CL_val, batch_size=16, shuffle=False,
                                           sampler=val_data_sampler, num_workers=5)

    start_epoch = 0
    for name, param in MBC_Networks.named_parameters():
        param.requires_grad = True
    # train loop

    lr = base_lr
    for epoch in range(start_epoch, start_epoch + reset_lr_epoch):
        # switch to train mode
        MBC_Networks.train()
        # lower learning rate
        if (epoch + 1) % reset_lr_epoch == 0:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        running_loss = 0.0
        count = 0

        if epoch < 1000:
            t_lr = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = t_lr
            for i, (cell_data, label) in enumerate(train_data):
                optimizer.zero_grad()
                cell_data = cell_data.to(device).float()
                label = label.to(device)
                cell_c, msm, att_f_i, att_f_l = MBC_Networks(cell_data)
                loss = CEloss(cell_c, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                count += 1
        print('epoch: ' + str(epoch) + '     loss: ' + str(running_loss / count) + '    learning rate: ' + str(lr))
        # ave_loss = running_loss / count

    # Validate
    # switch to test mode
    MBC_Networks.eval()
    TNeg = [0] * 21
    correct = [0] * 21
    count = [0] * 21
    pcount = [0.00001] * 21

    for i, (cell_data, label) in enumerate(val_data):
        cell_data = cell_data.to(device).float()
        label = label.to(device)
        cell_class, _, att_f_i, att_f_l = MBC_Networks(cell_data)
        cell_class = torch.softmax(cell_class, dim=-1)

        pred = cell_class.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_in = pred.eq(label.view_as(pred)).byte()[:, 0]
        for k, p, c in zip(label, pred, correct_in):
            k = k.item()
            p = p.item()
            c = c.item()
            count[k] += 1
            pcount[p] += 1
            correct[k] += c
            list_1 = [1] * 21
            list_1[k] = 0
            list_1[p] = 0
            TNeg = [x+y for x, y in zip(TNeg, list_1)]

    return count, pcount, correct, TNeg

