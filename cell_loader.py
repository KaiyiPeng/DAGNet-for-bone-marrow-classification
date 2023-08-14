import os
import torch
import random
import ast
import math
import cv2
import numpy as np
from PIL import Image, ImageFile
from collections import Counter
import torch.utils.data as data_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n

class Cell_PathLoader(torch.utils.data.Dataset):
    def __init__(self, cell_list_path):
        self.cell_list_path = cell_list_path

        cell_list_file = open(cell_list_path, 'r')
        self.cell_list_str = cell_list_file.readlines()

    def Load_Cell_list(self, folder_num=5, folder=0):
        cell_dct_list = []
        cell_cls_list = []
        cls_train_s1_list = []
        cls_train_e1_list = []
        cls_train_s2_list = []
        cls_train_e2_list = []
        cls_val_s_list = []
        cls_val_e_list = []
        train_cell_list = []
        val_cell_list = []
        for cell_dct_str in self.cell_list_str:
            cell_dct = ast.literal_eval(cell_dct_str)
            cell_dct_list.append(cell_dct)
            cell_cls_list.append(cell_dct['class'])
        cls_c_list = Counter(cell_cls_list)
        now_start = 0
        for cls_c in cls_c_list.values():
            folder_in_list = split_integer(cls_c, folder_num)
            folder_in_num = folder_in_list[folder]
            cls_train_s1_list.append(now_start)
            cls_train_e1_list.append(cls_train_s1_list[-1] + sum(folder_in_list[:folder]))
            cls_val_s_list.append(cls_train_e1_list[-1])
            cls_val_e_list.append(cls_val_s_list[-1] + folder_in_num)
            cls_train_s2_list.append(cls_val_e_list[-1])
            cls_train_e2_list.append(now_start + cls_c)
            now_start = cls_train_e2_list[-1]
        for i in range(len(cls_c_list)):
            train_cell_list.extend(cell_dct_list[cls_train_s1_list[i]:cls_train_e1_list[i]])
            train_cell_list.extend(cell_dct_list[cls_train_s2_list[i]:cls_train_e2_list[i]])
            val_cell_list.extend(cell_dct_list[cls_val_s_list[i]:cls_val_e_list[i]])

        c_id = None
        class_num = []
        class_list = []
        for cell_data in train_cell_list:
            cell_class = cell_data['class']
            class_list.append(cell_class)
            if c_id != cell_class:
                c_id = cell_class
                class_num.append(0)
                class_num[-1] = class_num[-1] + 1
            else:
                class_num[-1] = class_num[-1] + 1
        weight_list = [1.0/x for x in class_num]
        cell_weight_list = [weight_list[i] for i in class_list]

        return train_cell_list, val_cell_list, cell_weight_list

    def __getitem__(self, cell_id):
        pass

    def __len__(self):
        pass

class CellLoader(torch.utils.data.Dataset):
    def __init__(self, cell_path_list, f_height=-1, f_width=-1):
        self.cell_path_list = cell_path_list
        self.height, self.width = f_height, f_width

    def __getitem__(self, cell_id):
        cell_dct = self.cell_path_list[cell_id]
        cell_path = cell_dct['path']
        cell_class = cell_dct['class']
        img = Image.open(cell_path)
        img_array = np.array(img)
        cell_img = torch.from_numpy(img_array)
        cell_img = cell_img.permute(2, 0, 1)

        return cell_img, cell_class

    def __len__(self):
        return len(self.cell_list)

class CellSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)
