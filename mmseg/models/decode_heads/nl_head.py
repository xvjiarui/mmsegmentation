import torch
from mmcv.cnn import NonLocal2d

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class NLHead(FCNHead):

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)

    def forward(self, inputs):
        """Forward function. Omitted Here"""
        pass
        
