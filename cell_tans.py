import os

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data_dir', type=str, default='')
args = parser.parse_args()
data_dir = args.data_dir   # '/DISK0/DATA base/archive/'
data_folders = os.listdir(data_dir)
cell_list = []
for df in data_folders:
  df_path = data_dir + df + '/'
  if os.path.isdir(df_path):
    cell_folders = os.listdir(df_path)
    cell_folders.sort()
    for cf in cell_folders:
      cf_path = df_path + cf + '/'
      for home, dirs, cells in os.walk(cf_path):
        for cell in cells:
          cell_path = home + '/' + cell
          cell_class = cell_folders.index(cf)
          cell_dct = {'path': cell_path, 'class': cell_class}
          cell_list.append(cell_dct)

save_file = open(data_dir+'cell_list', 'w')
for fp in cell_list:
  save_file.write(str(fp))
  save_file.write('\n')
save_file.close()