import os 

class Seg:
    images = "data/data_seg/dataset"
    masks = "data/data_seg/groundtruth/mask"
    batch = 4
    lr = 1e-4
    epochs = 20
    model_dir = "./results/checkpoints"

class Config:
    learning_rate=1e3