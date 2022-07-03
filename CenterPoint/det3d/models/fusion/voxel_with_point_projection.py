import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.models.registry import FUSION
from det3d.models.model_utils.basic_block_1d import BasicBlock1D
from det3d.models.model_utils.actr import build as build_actr
from .point_to_image_projection import Point2ImageProjection

@FUSION.register_module
class VoxelWithPointProjection(nn.Module):
    def __init__(self, fuse_mode, interpolate, voxel_size, pc_range, image_list, image_scale=1,
                 depth_thres=0, double_flip=False, layer_channel=None, pfat_cfg=None, lt_cfg=None,
                 model_name='ACTR'
                 ):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.point_projector = Point2ImageProjection(voxel_size=voxel_size,
                                                     pc_range=pc_range,
                                                     depth_thres=depth_thres,
                                                     double_flip=double_flip)
        self.fuse_mode = fuse_mode
        self.image_interp = interpolate
        self.image_list = image_list
        self.image_scale = image_scale
        self.double_flip = double_flip
        if self.fuse_mode == 'concat':
            self.fuse_blocks = nn.ModuleDict()
            for _layer in layer_channel.keys():
                block_cfg = {"in_channels": layer_channel[_layer]*2,
                             "out_channels": layer_channel[_layer],
                             "kernel_size": 1,
                             "stride": 1,
                             "bias": False}
                self.fuse_blocks[_layer] = BasicBlock1D(**block_cfg)
        if self.fuse_mode == 'pfat':
            self.pfat = build_actr(pfat_cfg, lt_cfg=lt_cfg, model_name=model_name)


    def fusion(self, image_feat, voxel_feat, image_grid, layer_name=None, point_inv=None, fuse_mode=None):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if fuse_mode == 'sum':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat += fuse_feat.permute(1,0)
        elif fuse_mode == 'mean':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0)) / 2
        elif fuse_mode == 'concat':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            concat_feat = torch.cat([fuse_feat, voxel_feat.permute(1,0)], dim=0)
            voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
            voxel_feat = voxel_feat.permute(1,0)
        else:
            raise NotImplementedError
        
        return voxel_feat

    def forward(self, batch_dict, encoded_voxel=None, layer_name=None, img_conv_func=None, fuse_mode=None, d_factor=None):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                voxel_coords: (N, 4), Voxel coordinates with B,Z,Y,X
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
            encoded_voxel: (N, C), Sparse Voxel featuress
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
            voxel_features: (N, C), Sparse Image voxel features
    """
        batch_size = len(batch_dict['image_shape'][self.image_list[0].lower()])
        if fuse_mode == 'pfat':
            img_feat_n = []
            img_grid_n = [[] for _ in range(batch_size)]
            v_feat_n = [[] for _ in range(batch_size)]
            point_inv_n = [[] for _ in range(batch_size)]
            mask_n = [[] for _ in range(batch_size)]
        for cam_key in self.image_list:
            cam_key = cam_key.lower()
            # Generate sampling grid for frustum volume
            projection_dict = self.point_projector(voxel_coords=encoded_voxel.indices.float(),
                                                   image_scale=self.image_scale,
                                                   batch_dict=batch_dict, 
                                                   cam_key=cam_key,
                                                   d_factor=d_factor
                                                   )

            # check 
            if encoded_voxel is not None:
                in_bcakbone = True
            else:
                in_bcakbone = False
                encoded_voxel = batch_dict['encoded_spconv_tensor']
            if not self.training and self.double_flip:
                batch_size = batch_size * 4
            for _idx in range(batch_size): #(len(batch_dict['image_shape'][cam_key])):
                _idx_key = _idx//4 if self.double_flip else _idx
                image_feat = batch_dict['img_feat'][layer_name+'_feat2d'][cam_key][_idx_key]
                if img_conv_func:
                    image_feat = img_conv_func(image_feat.unsqueeze(0))[0]
                raw_shape = tuple(batch_dict['image_shape'][cam_key][_idx_key].cpu().numpy())
                feat_shape = image_feat.shape[-2:]
                if self.image_interp:
                    image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape[:2], mode='bilinear')[0]
                index_mask = encoded_voxel.indices[:,0]==_idx
                voxel_feat = encoded_voxel.features[index_mask]
                image_grid = projection_dict['image_grid'][_idx]
                voxel_grid = projection_dict['batch_voxel'][_idx]
                point_mask = projection_dict['point_mask'][_idx]
                image_depth = projection_dict['image_depths'][_idx]
                point_inv = projection_dict['point_inv'][_idx]
                # temporary use for validation
                # point_mask[len(voxel_feat):] -> 0 for batch construction
                voxel_mask = point_mask[:len(voxel_feat)]
                if self.training and 'overlap_mask' in batch_dict.keys():
                    overlap_mask = batch_dict['overlap_mask'][_idx]
                    is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
                    if 'depth_mask' in batch_dict.keys():
                        depth_mask = batch_dict['depth_mask'][_idx]
                        depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
                        is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
                        is_overlap = is_overlap & (~is_inrange)

                    image_grid = image_grid[~is_overlap]
                    voxel_grid = voxel_grid[~is_overlap]
                    point_mask = point_mask[~is_overlap]
                    point_inv = point_inv[~is_overlap]
                    voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
                if not self.image_interp:
                    image_grid = image_grid.float()
                    image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
                    image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
                    image_grid = image_grid.long()

                if fuse_mode == 'pfat':
                    img_feat_n.append(image_feat)
                    img_grid_n[_idx].append(image_grid[point_mask])
                    point_inv_n[_idx].append(point_inv[point_mask])
                    v_feat_n[_idx].append(voxel_feat[voxel_mask])
                    mask_n[_idx].append(voxel_mask)
                else:
                    voxel_feat[voxel_mask] = self.fusion(image_feat, voxel_feat[voxel_mask], 
                                                         image_grid[point_mask], layer_name, 
                                                         point_inv[point_mask], fuse_mode=fuse_mode)
                    encoded_voxel.features[index_mask] = voxel_feat
        if fuse_mode  == 'pfat':
            # 6*b, c, w, h -> b*6, c, w, h
            img_feat_n = torch.stack(img_feat_n)
            c, w, h = img_feat_n.shape[1:]
            img_feat_n = img_feat_n.reshape(6, batch_size, c, w, h)
            img_feat_n = img_feat_n.transpose(1, 0).reshape(-1, c, w, h)

            # aggregate
            max_ne = max([img_grid_n[b][i].shape[0] for b in range(batch_size) for i in range(6)])
            v_channel = voxel_feat.shape[1]
            img_grid_b = torch.zeros((batch_size*6, max_ne, 2)).cuda()
            pts_inv_b = torch.zeros((batch_size*6, max_ne, 3)).cuda()
            v_feat_b = torch.zeros((batch_size*6, max_ne, v_channel)).cuda()
            for b in range(batch_size):
                for i in range(6):
                    ne = img_grid_n[b][i].shape[0]
                    img_grid_b[b*6+i, :ne] = img_grid_n[b][i]
                    pts_inv_b[b*6+i, :ne] = point_inv_n[b][i]
                    v_feat_b[b*6+i, :ne] = v_feat_n[b][i]

            # for visualize
            if False:
                import cv2
                cam_list = list(batch_dict['images'].keys())
                for b in range(batch_size):
                    for i in range(6):
                        img_feat = img_feat_n[b*6+i].max(0)[0].detach().cpu()
                        img_feat_norm = ((img_feat - img_feat.min()) / (img_feat.max() - img_feat.min()) * 255).to(torch.uint8).numpy()
                        img_feat_norm_jet = cv2.applyColorMap(img_feat_norm, cv2.COLORMAP_JET)
                        cv2.imwrite('./vis/%06d_feat.png' % (b*6+i), img_feat_norm_jet)

                        img_grid = img_grid_b[b*6+i].to(torch.uint8).detach().cpu().numpy()
                        img_feat_norm[img_grid[:, 1], img_grid[:, 0]] = 255
                        img_feat_norm_jet = cv2.applyColorMap(img_feat_norm, cv2.COLORMAP_JET)
                        cv2.imwrite('./vis/%06d_feat_proj.png' % (b*6+i), img_feat_norm_jet)

                        img_o = batch_dict['images'][cam_list[i]][b]
                        img_o = ((img_o - img_o.min()) / (img_o.max() - img_o.min()) * 255).to(torch.uint8).numpy()
                        cv2.imwrite('./vis/%06d_original.png' % (b*6+i), img_o)
                import pdb; pdb.set_trace()

            img_grid_b /= torch.tensor(feat_shape[::-1]).cuda()
            enh_feat = self.pfat(v_feat=v_feat_b, grid=img_grid_b, i_feats=[img_feat_n], 
                                 lidar_grid=pts_inv_b)

            # split
            st = 0
            for b in range(batch_size):
                n_ne = (encoded_voxel.indices[:, 0] == b).sum()
                for i in range(6):
                    mask = mask_n[b][i]
                    num_ne = mask.nonzero().shape[0]
                    # for now fuse by sum
                    encoded_voxel.features[st:st+n_ne][mask] += enh_feat[b*6+i][:num_ne]
                st += n_ne

            return encoded_voxel
        else:
            return encoded_voxel

