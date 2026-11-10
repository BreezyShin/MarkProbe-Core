import math

import torch
from torch import nn
import torch.nn.functional as F


class SamePadConv2d(nn.Module):
    """Conv2d with TensorFlow/Keras ``padding='same'`` semantics."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 initializer="glorot"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size,
                              stride=self.stride, padding=0, bias=True)
        if initializer == "he":
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        out_h = math.ceil(in_h / self.stride[0])
        out_w = math.ceil(in_w / self.stride[1])
        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - in_h, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - in_w, 0)
        # TensorFlow SAME puts the extra pixel on bottom/right.
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2))
        return self.conv(x)


class ReluConv(nn.Module):
    """A single Conv2D + ReLU as used in the released StegaStamp model."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 initializer="he"):
        super().__init__()
        self.conv = SamePadConv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, initializer=initializer)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


def _upsample2x(x):
    # Keras UpSampling2D defaults to nearest-neighbour interpolation.
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Encoder(nn.Module):
    """PyTorch port of StegaStampEncoder (32/64/128/256 U-Net channels)."""

    def __init__(self, message_bits=96, width=32, image_size=400, residual_scale=0.1):
        super().__init__()
        if int(width) != 32:
            raise ValueError("StegaStamp encoder channel widths are fixed; width must be 32")
        self.message_bits = int(message_bits)
        self.image_size = int(image_size)
        self.secret_grid = 50
        # Kept as metadata/API compatibility only. StegaStamp predicts an unscaled,
        # linear RGB residual (there is no tanh or residual multiplier in the source).
        self.residual_scale = float(residual_scale)

        self.secret_dense = nn.Linear(self.message_bits, 7500)
        nn.init.kaiming_normal_(self.secret_dense.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.secret_dense.bias)
        self.secret_activation = nn.ReLU(inplace=True)

        self.conv1 = ReluConv(6, 32, 3, initializer="he")
        self.conv2 = ReluConv(32, 32, 3, stride=2, initializer="he")
        self.conv3 = ReluConv(32, 64, 3, stride=2, initializer="he")
        self.conv4 = ReluConv(64, 128, 3, stride=2, initializer="he")
        self.conv5 = ReluConv(128, 256, 3, stride=2, initializer="he")

        self.up6 = ReluConv(256, 128, 2, initializer="he")
        self.conv6 = ReluConv(256, 128, 3, initializer="he")
        self.up7 = ReluConv(128, 64, 2, initializer="he")
        self.conv7 = ReluConv(128, 64, 3, initializer="he")
        self.up8 = ReluConv(64, 32, 2, initializer="he")
        self.conv8 = ReluConv(64, 32, 3, initializer="he")
        self.up9 = ReluConv(32, 32, 2, initializer="he")
        self.conv9 = ReluConv(70, 32, 3, initializer="he")
        self.conv10 = ReluConv(32, 32, 3, initializer="he")
        self.residual = SamePadConv2d(32, 3, 1, initializer="he")

    def forward(self, image, message):
        if message.ndim != 2 or message.shape[1] != self.message_bits:
            raise ValueError("message must have shape (batch, {})".format(self.message_bits))
        if image.shape[0] != message.shape[0]:
            raise ValueError("image and message batch sizes must match")

        height, width = image.shape[-2:]
        secret = self.secret_activation(self.secret_dense(message - 0.5))
        secret = secret.view(image.shape[0], 3, self.secret_grid, self.secret_grid)
        secret = _upsample2x(_upsample2x(_upsample2x(secret)))
        # At StegaStamp's reference size (400x400), this is the exact x8 map.
        # Keep the same trained layer shapes at the core's lower-resolution smoke sizes.
        if secret.shape[-2:] != (height, width):
            secret = F.interpolate(secret, size=(height, width), mode="nearest")
        source = torch.cat((secret, image - 0.5), dim=1)

        conv1 = self.conv1(source)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        up6 = self.up6(_upsample2x(conv5))
        if up6.shape[-2:] != conv4.shape[-2:]:
            up6 = F.interpolate(up6, size=conv4.shape[-2:], mode="nearest")
        conv6 = self.conv6(torch.cat((conv4, up6), dim=1))

        up7 = self.up7(_upsample2x(conv6))
        if up7.shape[-2:] != conv3.shape[-2:]:
            up7 = F.interpolate(up7, size=conv3.shape[-2:], mode="nearest")
        conv7 = self.conv7(torch.cat((conv3, up7), dim=1))

        up8 = self.up8(_upsample2x(conv7))
        if up8.shape[-2:] != conv2.shape[-2:]:
            up8 = F.interpolate(up8, size=conv2.shape[-2:], mode="nearest")
        conv8 = self.conv8(torch.cat((conv2, up8), dim=1))

        up9 = self.up9(_upsample2x(conv8))
        if up9.shape[-2:] != conv1.shape[-2:]:
            up9 = F.interpolate(up9, size=conv1.shape[-2:], mode="nearest")
        merge9 = torch.cat((conv1, up9, source), dim=1)
        conv9 = self.conv9(merge9)

        # Keep both released source layers and its output tap: the reference
        # models.py computes conv10(conv9) but feeds conv9 to the residual head.
        _conv10 = self.conv10(conv9)
        return self.residual(conv9)


class SpatialTransformer(nn.Module):
    """StegaStamp's localization CNN and identity-initialized affine STN."""

    def __init__(self, height=128, width=128, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        if not self.enabled:
            self.localization = None
            self.fc = None
            return

        self.height = int(height)
        self.width = int(width)
        h, w = self.height, self.width
        for _ in range(3):
            h, w = math.ceil(h / 2), math.ceil(w / 2)
        localization_fc = nn.Linear(128 * h * w, 128)
        nn.init.xavier_uniform_(localization_fc.weight)
        nn.init.zeros_(localization_fc.bias)
        self.localization = nn.Sequential(
            ReluConv(3, 32, 3, stride=2, initializer="glorot"),
            ReluConv(32, 64, 3, stride=2, initializer="glorot"),
            ReluConv(64, 128, 3, stride=2, initializer="glorot"),
            nn.Flatten(),
            localization_fc,
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128, 6)
        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.fc.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, image):
        if not self.enabled:
            return image
        if image.shape[-2:] != (self.height, self.width):
            image = F.interpolate(image, size=(self.height, self.width),
                                  mode="bilinear", align_corners=False)
        theta = self.fc(self.localization(image)).view(-1, 2, 3)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)


class Decoder(nn.Module):
    """StegaStamp STN + seven-convolution secret decoder."""

    def __init__(self, message_bits=96, width=32, use_stn=True,
                 image_size=400, image_width=None):
        super().__init__()
        if int(width) != 32:
            raise ValueError("StegaStamp decoder channel widths are fixed; width must be 32")
        self.image_size = (int(image_size), int(image_width or image_size))
        self.stn = SpatialTransformer(*self.image_size, enabled=use_stn)

        self.conv1 = ReluConv(3, 32, 3, stride=2, initializer="glorot")
        self.conv2 = ReluConv(32, 32, 3, initializer="glorot")
        self.conv3 = ReluConv(32, 64, 3, stride=2, initializer="glorot")
        self.conv4 = ReluConv(64, 64, 3, initializer="glorot")
        self.conv5 = ReluConv(64, 64, 3, stride=2, initializer="glorot")
        self.conv6 = ReluConv(64, 128, 3, stride=2, initializer="glorot")
        self.conv7 = ReluConv(128, 128, 3, stride=2, initializer="glorot")

        h, w = self.image_size
        for _ in range(5):
            h, w = math.ceil(h / 2), math.ceil(w / 2)
        self.fc1 = nn.Linear(128 * h * w, 512)
        self.fc2 = nn.Linear(512, int(message_bits))
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, image):
        # StegaStamp's dense decoder has a fixed flatten dimension determined
        # by training resolution; normalize arbitrary inference sizes to it.
        if image.shape[-2:] != self.image_size:
            image = F.interpolate(image, size=self.image_size,
                                  mode="bilinear", align_corners=False)
        image = self.stn(image - 0.5)
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)


class MarkProbe(nn.Module):
    """StegaStamp-style image/message encoder and robust bit-logit decoder."""

    def __init__(self, message_bits=96, width=32, use_stn=True,
                 residual_scale=0.1, image_size=400, image_width=None):
        super().__init__()
        if int(width) != 32:
            raise ValueError("The StegaStamp architecture uses fixed 32/64/128/256 channels; width must be 32")
        self.message_bits = int(message_bits)
        self.width = 32
        self.use_stn = bool(use_stn)
        self.residual_scale = float(residual_scale)  # legacy metadata; not applied
        self.image_size = (int(image_size), int(image_width or image_size))
        self.encoder = Encoder(message_bits, width=32, image_size=self.image_size[0],
                               residual_scale=residual_scale)
        self.decoder = Decoder(message_bits, width=32, use_stn=use_stn,
                               image_size=self.image_size[0], image_width=self.image_size[1])

    def encode(self, image, message):
        if message.ndim != 2 or message.shape != (image.shape[0], self.message_bits):
            raise ValueError("message must have shape (batch, {})".format(self.message_bits))
        if image.shape[-2:] != self.image_size:
            image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        return image + self.encoder(image, message)

    def decode(self, image):
        return self.decoder(image)

    def forward(self, image, message):
        encoded = self.encode(image, message)
        return encoded, self.decode(encoded)
