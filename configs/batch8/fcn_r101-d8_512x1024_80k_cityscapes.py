_base_ = '../fcn/fcn_r50-d8_512x1024_80k_cityscapes.py'
norm_cfg = dict(type='MMSyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101, norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))
data = dict(samples_per_gpu=1, workers_per_gpu=2)
