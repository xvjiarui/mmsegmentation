_base_ = './pspnet_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet50_1by4_v1c',
    backbone=dict(stem_channels=16, base_channels=16),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
