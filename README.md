# DAGNet-for-bone-marrow-classification
daul attention gates dense neural networks(DAGNet) for bone marrow classification

We propose DAGNet based on DenseNet for classifying cells in bone marrow smears. The classification model training and testing data sets come from Munich Leukemia Laboratory (MLL), the address is: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770

After downloading and decompressing the MLL data set, execute the following command to generate a list of cell image paths

```cpp
python cell_tans.py --data_dir=''
```

"data_dir" is the path where the "bone_marrow_cell_dataset" folder is located.

```cpp
The command for five-folders cross-validation is
```

python train_5f.py --model=DAGNet --resume=True --cell_list_path='./archive/cell_list'

The "model" parameter is used to select different network models, and the optional models include densenet, DAGNet, resNeXt50, resNeXt101. The "resume" parameter is the option to load the pre-trained model. "cell_list_path" indicates the address list file of all images in the dataset.
