import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaFCS.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from MambaFCS.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from MambaFCS.changedetection.models.GuidedFusion import PyramidFusion, DepthwiseSeparableConv, FFT_Fusion


def _make_st_block(hidden_dim, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, kwargs):
    """Un VSSBlock encadré des permutations channel-last <-> channel-first."""
    return nn.Sequential(
        Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
        VSSBlock(hidden_dim=hidden_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
        Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
    )


class ChangeDecoder(nn.Module):
    """Décodeur de détection de changement binaire (BCD).

    Généralisé à un nombre arbitraire de stages d'encodeur : la profondeur est
    déduite de ``len(encoder_dims)``. Les stages sont indexés du plus superficiel
    (index 0, haute résolution) au plus profond (index N-1, basse résolution),
    dans le même ordre que la sortie de l'encodeur. Retirer un stage à l'encodeur
    (ex. le plus profond) ne demande donc aucune modification ici.

    Pour N = 4 le graphe est strictement identique à l'implémentation d'origine.
    """

    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()
        self.num_stages = len(encoder_dims)

        # Un bloc VSS + une fusion spatio-temporelle par stage.
        self.st_blocks = nn.ModuleList([
            _make_st_block(encoder_dims[i], channel_first, norm_layer, ssm_act_layer, mlp_act_layer, kwargs)
            for i in range(self.num_stages)
        ])
        self.fuse_layers = nn.ModuleList([
            FFT_Fusion(in_channels=encoder_dims[i], use_diff=True)
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

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def _down(self, i, x):
        # down_samples est indexée à partir du stage 1 : le stage i utilise down_samples[i-1].
        return self.down_samples[i - 1](x)

    def forward(self, pre_features, post_features):
        N = self.num_stages
        attentions = [None] * N

        # ----- Stage le plus profond -----
        i = N - 1
        p = self.fuse_layers[i](pre_features[i], post_features[i])
        p = self.st_blocks[i](p)
        attentions[i] = p
        p = self._down(i, p)

        # ----- Stages superficiels (profond -> superficiel) -----
        for i in range(N - 2, -1, -1):
            f = self.fuse_layers[i](pre_features[i], post_features[i])
            f = self._upsample_add(p, f)
            f = self.smooth_layers[i](f)
            f = self.st_blocks[i](f)
            attentions[i] = f
            if i > 0:
                p = self._down(i, f)
            else:
                p = f

        # change_maps ordonnées du plus profond au plus superficiel
        # (comportement identique à l'ancien [p4, p3, p2, p1]).
        change_maps = [attentions[i] for i in range(N - 1, -1, -1)]
        return p, change_maps
