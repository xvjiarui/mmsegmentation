_base_ = './pspnet_r50-d8_512x1024_80k_cityscapes.py'
model = dict(type='ResNet', pretrained='torchvision://resnet50')
