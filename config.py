import tensorkit as tk

config = tk.Config()
_config = {
    'TRAIN_DS': './dataset/kalantari_dataset/train',
    'TRAIN_SIZE': 74,
    'VAL_DS': './dataset/kalantari_dataset/val',
    'VAL_SIZE': 55,
    'VAL': False,
    'TEST_DS': './dataset/kalantari_dataset/test',
    'TEST_SIZE': 15,

    'train_hw': (512, 512),  # The size of images for training
    'val_hw': (256, 256),
    'test_hw': (960, 1440),

    'BATCH_SIZE': 6,
    'EPOCH': 256000,

    'LR': 1e-4,
    'LR_DECAY': 0.92,  # per epoch from 90 epoch to 120 epoch
    'MOMENTUM': 0.9,

    'LOG_DIR': './logs',
    'TEST_DIR': './results',

    'SAVE_STEP': 1000,
    'SUMMARY_STEP': 200,

    'CUDA_VISIBLE_DEVICES': [0],
    'ALLOW_GROWTH': False,

    'BACKUP_DIR': ['model', 'tensorkit'],
    'VGG_CKPT': './model_zoo/vgg_16.ckpt',
}

config.apply(_config)

if __name__ == '__main__':
    config.show()
