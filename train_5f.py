import os
import torch
from train_func import train_func
from prettytable import PrettyTable
import warnings

import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--model', type=str, default='densenet')
parser.add_argument('--resume', type=str, default='True')
args = parser.parse_args()

model_name = args.model  #DAGNet densenet resNeXt101 resNeXt50
resume = args.resume == 'True'
base_lr = 0.001
batch_size = 32
folder_num = 5
reset_lr_epoch = 5
cell_list_path = '/DISK0/DATA base/archive/cell_list'

t_correct = [0] * 21
t_count = [0] * 21
t_pcount = [0] * 21
for f in range(folder_num):
    count, pcount, correct, TNeg = train_func(model_name=model_name, resume=resume, base_lr=base_lr, batch_size=batch_size,
                                              reset_lr_epoch=reset_lr_epoch, folder_num=folder_num, folder=f,
                                              cell_list_path=cell_list_path)
    t_count = [t_count[i] + count[i] for i in range(0, len(t_count))]
    t_pcount = [t_pcount[i] + pcount[i] for i in range(0, len(pcount))]
    t_correct = [t_correct[i] + correct[i] for i in range(0, len(t_correct))]
    rec_list = [correct[i] / count[i] for i in range(0, len(correct))]
    pre_list = [correct[i] / pcount[i] for i in range(0, len(correct))]
    s_count = sum(count)
    s_correct = sum(correct)
    table_x = PrettyTable()
    table_x.field_names = ["", "Abnormal eosinophils", "Artefacts", "Basophils", "Blasts", "Erythroblasts", "Eosinophils"
                           , "Faggot cells", "Hairy cells", "Smudge cells", "Immature lymphocytes", "Lymphocytes"
                           , "Metamyelocytes", "Monocytes", "Myelocytes", "Band neutrophils", "Segmented neutrophils"
                           , "Not identifiable", "Other cells", "Promyelocytes", "Plasma cells", "Proerythroblasts"]
    rec_list_3 = ['{:.3f}'.format(x) for x in rec_list]
    pre_list_3 = ['{:.3f}'.format(x) for x in pre_list]
    table_x.add_row(['Recall'] + rec_list_3)
    table_x.add_row(['Precision'] + pre_list_3)
    print(table_x)
    print('Test correct/test count: %s/%s AVE Precision per case: %.3f \n' % (s_correct, s_count, 100. * s_correct / s_count))

t_rec_list = [t_correct[i] / t_count[i] for i in range(0, len(t_correct))]
t_pre_list = [t_correct[i] / t_pcount[i] for i in range(0, len(t_correct))]
t_s_count = sum(t_count)
t_s_correct = sum(t_correct)
table_tx = PrettyTable()
table_tx.field_names = ["MEAN", "Abnormal eosinophils", "Artefacts", "Basophils", "Blasts", "Erythroblasts", "Eosinophils"
                        , "Faggot cells", "Hairy cells", "Smudge cells", "Immature lymphocytes", "Lymphocytes"
                        , "Metamyelocytes", "Monocytes", "Myelocytes", "Band neutrophils", "Segmented neutrophils"
                        , "Not identifiable", "Other cells", "Promyelocytes", "Plasma cells", "Proerythroblasts"]
t_rec_list_3 = ['{:.3f}'.format(x) for x in t_rec_list]
t_pre_list_3 = ['{:.3f}'.format(x) for x in t_pre_list]
table_tx.add_row(['Recall'] + t_rec_list_3)
table_tx.add_row(['Precision'] + t_pre_list_3)
print(table_tx)
print('Mean correct/total: %s/%s AVE Precision per case: %.3f \n' % (t_s_correct, t_s_count, 100. * t_s_correct / t_s_count))

