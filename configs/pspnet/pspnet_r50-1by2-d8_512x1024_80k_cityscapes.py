_base_ = './pspnet_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet50_1by2_v1c',
    backbone=dict(stem_channels=32, base_channels=32),
    decode_head=dict(
        in_channels=1024,
        channels=256,
    ),
    auxiliary_head=dict(in_channels=512, channels=128))
