# Anonymous Submission ID 1458 for MM-2020

### Requirements
- Python 3.6
- Pytorch 1.3


### Datasets
The links of datasets will be released afterwards,
- Office-Caltech
- NUSWIDE-ImageNet
- Multilingual Reuters Collection


### Training
The general command for training is,
```
python3 train.py
```
Change arguments for different experiments:
- dataset: "office" / "nusimg" / "mrc"
- batch_size: mini_batch size
- beta: The ratio of known target sample and Unk target sample in the pseudo label set
- num_layers: GNN's depth
- edge_loss: edge classification loss
- dis_loss: discrepancy loss
- c_loss: clustering embedding loss

For the detailed hyper-parameters setting for each dataset, please refer to Section 5.2 and Appendix 3.  

Remember to change dataset_root to suit your own case

The training loss and validation accuracy will be automatically saved in './logs/', which can be visualized with tensorboard.
The model weights will be saved in './checkpoints'
