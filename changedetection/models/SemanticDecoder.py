import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaFCS.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from MambaFCS.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from MambaFCS.changedetection.models.GuidedFusion import PyramidFusion, FFTBranch
from MambaFCS.changedetection.models.MultiScaleChangeGuidedAttention import ChangeGuidedAttention
from MambaFCS.changedetection.models.ChangeDecoder import _make_st_block

import os
main_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))


class SemanticDecoder(nn.Module):
    """Décodeur de segmentation sémantique (T1 ou T2), guidé par les cartes de changement.

    Généralisé à un nombre arbitraire de stages (déduit de ``len(encoder_dims)``).
    ``features`` et ``change_maps`` sont ordonnés du plus superficiel (index 0) au
    plus profond (index N-1). Pour N = 4 le graphe est strictement identique à
    l'implémentation d'origine.
    """

    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(SemanticDecoder, self).__init__()
        self.num_stages = len(encoder_dims)
        self.CHANGE_GUIDED_ATTENTION = True

        self.st_blocks = nn.ModuleList([
            _make_st_block(encoder_dims[i], channel_first, norm_layer, ssm_act_layer, mlp_act_layer, kwargs)
            for i in range(self.num_stages)
        ])
        # Projection de canaux stage i -> stage i-1 (pour i = 1..N-1).
        self.down_samples = nn.ModuleList([
            PyramidFusion(in_channels=encoder_dims[i], out_channels=encoder_dims[i - 1])
            for i in range(1, self.num_stages)
        ])
        # Lissage des stages superficiels qui reçoivent un upsample-add (i = 0..N-2).
        self.smooth_layers = nn.ModuleList([
            ResBlock(in_channels=encoder_dims[i], out_channels=encoder_dims[i], stride=1)
            for i in range(self.num_stages - 1)
        ])
        # Lissage final appliqué à la sortie la plus superficielle.
        self.final_smooth = ResBlock(in_channels=encoder_dims[0], out_channels=encoder_dims[0], stride=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def _guided(self, feat, change_map):
        if self.CHANGE_GUIDED_ATTENTION:
            return ChangeGuidedAttention()(feat, change_map)
        return feat

    def forward(self, features, change_maps):
        N = self.num_stages

        # ----- Stage le plus profond -----
        i = N - 1
        p = self._guided(features[i], change_maps[i])
        p = self.st_blocks[i](p)
        p = self.down_samples[i - 1](p)

        # ----- Stages superficiels (profond -> superficiel) -----
        for i in range(N - 2, -1, -1):
            f = self._guided(features[i], change_maps[i])
            f = self._upsample_add(p, f)
            f = self.smooth_layers[i](f)
            f = self.st_blocks[i](f)
            if i > 0:
                p = self.down_samples[i - 1](f)
            else:
                p = f

        p = self.final_smooth(p)
        return p
