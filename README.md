# DAGNet-for-bone-marrow-classification
daul attention gates dense neural networks(DAGNet) for bone marrow classification

We propose DAGNet based on DenseNet for classifying cells in bone marrow smears. The classification model training and testing data sets come from Munich Leukemia Laboratory (MLL), the address is: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770

The command for five-point cross-validation is
python train_5f.py --model=DAGNet --resume=True

The "model" parameter is used to select different network models, and the optional models include densenet, DAGNet, resNeXt50, resNeXt101. The "resume" parameter is the option to load the pre-trained model.
