import torch
import torch.nn as nn

from ..vfe.image_vfe_modules.ffn.ddn_loss.balancer import BalancerACTR as Balancer
from pcdet.utils import transform_utils, loss_utils
import numpy as np

try:
    from kornia.losses.focal import FocalLoss
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


class SEGLOSS(nn.Module):

    def __init__(self, weight, alpha, gamma, fg_weight, bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha,
                                   gamma=self.gamma,
                                   reduction="none")
        self.weight = weight
        use_conv_for_no_stride = None
        upsample_strides = [0.5, 1, 2, 4]
        deblocks = []
        in_channels = 256
        out_channel = 256
        self.num_classes = 1  # foreground : 1 background : 0
        self.downsample_factor = downsample_factor
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        for i in range(len(upsample_strides)):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])

            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = torch.nn.Upsample(
                    scale_factor=upsample_strides[i])
                # upsample_layer = torch.nn.Upsample(input = img_aux_feats, scale_factor=upsample_strides[i])
            deblock = nn.Sequential(upsample_layer,
                                    torch.nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.aux_img_cls_agg = nn.ModuleList(deblocks)
        self.aux_img_cls = nn.Conv2d(out_channel, self.num_classes, 1)

    def pred_fg_img(self, batch_dict):

        cat_list = []
        aux_img_losses_cls = torch.tensor(0.).cuda()
        batch_size = batch_dict['batch_size']
        img_aux_feats = batch_dict['img_dict']

        for i in range(len(img_aux_feats)):
            if i == 0: img_aux_feats[0] = img_aux_feats['layer1_feat2d']
            cat_list.append(self.aux_img_cls_agg[i](img_aux_feats[i]))
        # cat_list.append(self.aux_img_cls_agg(img_aux_feats))

        cat_feat = torch.cat(cat_list, dim=1)

        pred_fg_img = self.aux_img_cls(cat_feat)
        target_size = self.aux_img_cls(cat_feat).shape
        # gt_cls_img = nn.functional.interpolate(
        #     img_mask.unsqueeze(1), size=pred_fg_img.shape[2:]).unsqueeze(dim=2)

        pred_fg_img_list = pred_fg_img.permute(0, 2, 3, 1).clone()
        pred_fg_img = pred_fg_img.permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.num_classes).squeeze(2).unsqueeze(1)

        # gt_cls_img = gt_cls_img.squeeze().reshape(batch_size, -1).squeeze(1).float()
        # for b in range(batch_size):
        #     aux_img_loss_cls = self.aux_img_loss_cls(
        #         pred_cls_img[b], gt_cls_img[b].to(torch.long))
        #     aux_img_losses_cls += aux_img_loss_cls
        return pred_fg_img, target_size

    def forward(self, batch_dict, tb_dict):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}
        gt_boxes2d = batch_dict['gt_boxes2d']
        images = batch_dict['images']
        images_rei = torch.zeros(images.shape[0], 1, images.shape[2],
                                 images.shape[3])

        fg_pred, target_size = self.pred_fg_img(batch_dict)
        target_size = torch.zeros(target_size[0], target_size[2],
                                  target_size[3])
        # Compute loss
        # loss = self.loss_func(depth_logits, depth_target)

        fg_mask = loss_utils.compute_fg_mask(
            gt_boxes2d=gt_boxes2d,
            shape=images_rei.shape,
            downsample_factor=self.downsample_factor,
            device=images_rei.device)
        fg_mask = nn.functional.interpolate(fg_mask.to(torch.float),
                                            size=target_size.shape[1:])
        fg_mask = fg_mask.permute(0, 2, 3, 1).reshape(
            batch_dict['batch_size'], -1,
            self.num_classes).squeeze(2).unsqueeze(1).to(torch.int64)

        # Compute loss
        loss = self.loss_func(fg_pred, fg_mask.squeeze(1).cuda())
        # Bin depth map to create target
        # depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, fg_mask=fg_mask)

        # Final loss
        loss *= self.weight
        tb_dict.update({"aux_seg_loss": loss.item()})

        return loss, tb_dict
