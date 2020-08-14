_base_ = './pspnet_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c_1by2',
    backbone=dict(depth=18, stem_channels=32, base_channels=32),
    decode_head=dict(
        in_channels=256,
        channels=56,
    ),
    auxiliary_head=dict(in_channels=128, channels=32))
