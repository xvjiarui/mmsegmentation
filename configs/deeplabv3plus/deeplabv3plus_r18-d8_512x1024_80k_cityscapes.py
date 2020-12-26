_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained=
    'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))