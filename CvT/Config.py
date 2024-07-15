import torch

config = {
    'model_name': 'microsoft/cvt-21',
    'n_labels': 14,
    'batch_size': 32,
    'dropout': 0.325,
    'hidden_size': 128,
    'lr': 7e-6,
    'n_epochs': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_img_train': 10000,
    'n_img_val': 1,
    'img_train': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\train',
    'img_val': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\val',
    'label_train':r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\fairface_label_train.csv',
    'label_val': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\fairface_label_val.csv',
    'img_train_path': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval',
    'reload': False,
    'img_train_rl': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\img_train_1_8.pt',
    'img_val_rl': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\img_val20.pt',
    'label_train_rl': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\labels_train_1_8.csv',
    'label_val_rl': r'C:\Users\rabby\CS 7643 - Deep Learning\Project\Data\fairface-img-margin025-trainval\labels_val20.csv'
}