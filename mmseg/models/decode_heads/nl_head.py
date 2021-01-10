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
        self.nl_block = NonLocal2d()

    def forward(self, inputs):
        """Forward function. Omitted Here"""
        pass
        
