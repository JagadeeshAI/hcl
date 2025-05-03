class Seg:
    images = "data/data_seg/dataset"
    masks = "data/data_seg/groundtruth/mask"
    batch = 4
    lr = 1e-4
    epochs = 2000
    model_dir = "./results/checkpoints"
    samples = "results/samples"
    scored = "results/scored"           # ← this is where output images go
    scores = "results/scores.json"      # ← this is the output JSON

class Config:
    learning_rate = 1e-3
