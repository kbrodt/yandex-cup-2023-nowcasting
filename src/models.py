import torch
import torch.nn as nn
import timm
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class TimmEncoder(nn.Module):
    def __init__(self, backbone, in_channels, out_indices, output_stride=32):
        super().__init__()

        depth = len(out_indices)
        self.model = timm.create_model(
            backbone,
            in_chans=in_channels,
            pretrained=True,
            num_classes=0,
            features_only=True,
            output_stride=output_stride if output_stride != 32 else None,
            out_indices=out_indices,
        )
        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride  # 32

    def forward(self, x):
        features = self.model(x)

        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


class UNet(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels,
        out_indices,
        dec_channels,
        dec_attn_type,
        n_classes,
        activation="relu",
        decoder_use_batchnorm: bool = True,
    ):
        super().__init__()

        encoder_name = backbone

        self.encoder = TimmEncoder(backbone, in_channels, out_indices)

        encoder_depth = len(self.encoder.out_channels) - 1

        decoder_channels = dec_channels[:encoder_depth]
        #enc_out_channels = self.encoder.out_channels
        enc_out_channels = self.encoder.out_channels[:-1] + [self.encoder.out_channels[-1] + 1]
        self.decoder = UnetDecoder(
            encoder_channels=enc_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=dec_attn_type,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], n_classes, kernel_size=3, padding=1)
        if activation == "relu":
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Identity()

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x, month=None):#, last_x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # x: [b, t, c, h, w]

        x = x.squeeze(2)  # [b, t, h, w]

        # 252 - 2*2= 256
        x = torch.nn.functional.pad(
            x,
            (2, 2, 2, 2),
            #mode="replicate",
        )

        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)
        features[-1] = torch.cat(
            [
                features[-1],
                nn.functional.interpolate(
                    month.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),  # [b] -> [b, 1, 1, 1]
                    size=features[-1].shape[-2:],
                    mode="nearest",
                ),
            ],
            dim=1,
        )

        features = [None] + features

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)   # [b, c, h, w]
        masks = masks[:, :, 2:-2, 2:-2]
        masks = self.relu(masks)

        masks = masks.unsqueeze(2)  # [b, c, 1, h, w]

        return masks
