import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_fpn import Encoder_FPN, SimpleEncoderDeep

from models.diff_ras.polygon import SoftPolygon

import torchvision
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

from torch_geometric.nn.conv import GCNConv
from utils.train_utils import sample_ellipse_fast
from utils.graph_utils import IVUS_Graph, Pool


class POLYCORE(nn.Module):
    """
    Encoder -> (BBox Regression Head, Polygon Segmentation Head)
    """

    def __init__(self, config):
        super().__init__()

        self.resolution = config['RESOLUTION']
        self.n_classes = config['N_CLASSES']
        self.image_dim = config['IMAGE_DIM']
        self.fpn_dim = config['FPN_DIM']
        self.graph_dim = config['GRAPH_DIM']
        self.device = config['DEVICE']
        self.num_points = config['NUM_POINTS']
        self.raster_size = config['RASTER_SIZE']
        self.loss = config['LOSS']
        self.num_levels = config['NUM_LEVELS']
        self.do_rasterize = config['RASTERIZE']

        self.adj_mat, self.up_mat, self.down_mat, self.edge_list = IVUS_Graph(inter=False)

        self.BACKBONE = self.create_backbone(config['BACKBONE'])
        self.BBOX_HEAD = nn.Sequential(
            nn.Conv2d(self.fpn_dim, 64, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*15*15, 1024), nn.ReLU(),
            nn.Linear(1024,128), nn.ReLU(),
            nn.Linear(128,8), nn.Sigmoid())
        self.POLYGON_HEAD = GCN(self.adj_mat, self.up_mat, self.graph_dim, self.num_levels, self.fpn_dim, self.edge_list)
        self.RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=0.1)

        self.REFINEMENT = Refiner()

    def forward(self, x):

        bs = x.shape[0]

        feature_maps = self.BACKBONE(x)

        bbox_pred = self.BBOX_HEAD(feature_maps[-1])

        ref_points = sample_ellipse_fast(
            0.5 * torch.ones((bs, 2), device=self.device),
            0.5 * torch.ones((bs, 2), device=self.device),
            0.5 * torch.ones((bs, 2), device=self.device),
            0.5 * torch.ones((bs, 2), device=self.device),
            count=self.num_points[0])

        bbox_pred = bbox_pred.reshape(bs, 2, 4)
        ref_scaled = self.scale_points(ref_points, bbox_pred)

        pts_pred = self.POLYGON_HEAD(feature_maps, ref_scaled)
        pts_pred = [o.reshape(bs, 2, -1, 2) for o in pts_pred]

        refined_pts, vector_field_pred, dist_pred = self.REFINEMENT(x, pts_pred[-1])

        if self.do_rasterize:
            refined_output = self.rasterize([refined_pts], [480])[0]
            return refined_output, refined_pts, vector_field_pred, dist_pred, [ref_scaled] + pts_pred, bbox_pred.reshape(bs, -1)
        else:
            return refined_pts, vector_field_pred, dist_pred, [ref_scaled] + pts_pred, bbox_pred.reshape(bs, -1)

    def create_backbone(self, name):

        if name == 'ResNet50FPN':
            encoder = resnet50(weights='DEFAULT')
            backbone = _resnet_fpn_extractor(encoder, trainable_layers=5)
        elif name == 'VanillaDeepFPN':
            encoder = SimpleEncoderDeep(input_dim=self.image_dim)
            fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 512, 512], self.fpn_dim)
            backbone = Encoder_FPN(encoder, fpn, levels=[3, 4, 5, 6])
        else:
            print('No model selected!')
        return backbone

    def scale_points(self, ref, bboxes):
        """
        scale points so they are in terms of image and not bbox.

        ref points: [BS, I, PTS, XY] in interval [0,1]
        bboxes: [BS, I, XYXY] in interval [0,1]

        returns - scaled_ref_points: [BS, I, PTS, XY]
        """
        num_pts = ref.shape[2]

        width = bboxes[:, :, 2] - bboxes[:, :, 0]
        height = bboxes[:, :, 3] - bboxes[:, :, 1]

        ref_x = (ref[:, :, :, 0] * width.unsqueeze(-1)) + bboxes[:, :, 0].unsqueeze(-1).repeat(1, 1, num_pts)
        ref_y = (ref[:, :, :, 1] * height.unsqueeze(-1)) + bboxes[:, :, 1].unsqueeze(-1).repeat(1, 1, num_pts)

        ref_scaled = torch.stack([ref_x, ref_y], -1)

        return ref_scaled

    def rasterize(self, pts, resolution, offset=-0.50):

        BS = pts[0].shape[0]

        rasters = []

        for i, (pts_lvl, res) in enumerate(zip(pts, resolution)):

            points_to_rasterize = torch.cat([pts_lvl[:, 0, :, :], pts_lvl[:, 1, :, :]], dim=0)
            raster = self.RASTERIZER(points_to_rasterize * float(res) + offset, res, res, 0.1)

            eem_raster = raster[:BS, :, :]
            lumen_raster = raster[BS:, :, :]

            eem_raster = eem_raster - lumen_raster
            background = torch.ones((pts_lvl.shape[0], res, res), device=pts_lvl.device)
            non_background = torch.clip(eem_raster + lumen_raster, 0, 1)
            background = background - non_background
            raster = torch.stack([background, eem_raster, lumen_raster], dim=1).to(torch.float32)


            rasters.append(raster)

        return rasters

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.InstanceNorm2d(output_dim))

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.InstanceNorm2d(output_dim))

        self.residual = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1))
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += res
        x = self.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, residual_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim + residual_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.InstanceNorm2d(output_dim))

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.InstanceNorm2d(output_dim))

        self.residual = nn.Conv2d(input_dim + residual_dim, output_dim, kernel_size=(1, 1))
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += res
        x = self.relu(x)
        return x

class Vector_Field(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_down_0 = EncoderBlock(3, 32)
        self.conv_down_1 = EncoderBlock(32, 64)
        self.conv_down_2 = EncoderBlock(64, 128)
        self.conv_down_3 = EncoderBlock(128, 256)
        self.conv_down_4 = EncoderBlock(256, 512)
        self.conv_down_5 = EncoderBlock(512, 512)

        self.bottleneck = EncoderBlock(512, 512)

        self.conv_up5 = DecoderBlock(512, 512, 512)
        self.conv_up4 = DecoderBlock(512, 512, 512)
        self.conv_up3 = DecoderBlock(512, 256, 256)
        self.conv_up2 = DecoderBlock(256, 128, 128)
        self.conv_up1 = DecoderBlock(128, 64, 64)
        self.conv_up0 = DecoderBlock(64, 32, 32)
        self.out_vec = nn.Sequential(nn.Conv2d(32, 4, kernel_size=3, padding=1), nn.Tanh())
        self.out_dist = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1), nn.Sigmoid())

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):

        residuals = []

        down0 = self.conv_down_0(x)
        residuals.append(down0)

        down1 = self.conv_down_1(down0)
        residuals.append(down1)
        down1_mp = self.maxpool(down1)

        down2 = self.conv_down_2(down1_mp)
        residuals.append(down2)
        down2_mp = self.maxpool(down2)

        down3 = self.conv_down_3(down2_mp)
        residuals.append(down3)
        down3_mp = self.maxpool(down3)

        down4 = self.conv_down_4(down3_mp)
        residuals.append(down4)
        down4_mp = self.maxpool(down4)

        down5 = self.conv_down_5(down4_mp)
        residuals.append(down5)
        down5_mp = self.maxpool(down5)

        bottleneck = self.bottleneck(down5_mp)

        up5 = torch.cat([self.upsample(bottleneck), down5], dim=1)
        up5 = self.conv_up5(up5)

        up4 = torch.cat([self.upsample(up5), down4], dim=1)
        up4 = self.conv_up4(up4)

        up3 = torch.cat([self.upsample(up4), down3], dim=1)
        up3 = self.conv_up3(up3)

        up2 = torch.cat([self.upsample(up3), down2], dim=1)
        up2 = self.conv_up2(up2)

        up1 = torch.cat([self.upsample(up2), down1], dim=1)
        up1 = self.conv_up1(up1)

        up0 = torch.cat([up1, down0], dim=1)
        up0 = self.conv_up0(up0)

        vector_field = self.out_vec(up0)
        dist_map = self.out_dist(up0)

        return vector_field, dist_map


def normalize(x, newRange=(0, 1)):  # x is an array. Default range is between zero and one
    xmin, xmax = torch.min(x), torch.max(x)  # get max and min from input array
    norm = (x - xmin) / (xmax - xmin)  # scale between zero and one

    if newRange == (0, 1):
        return (norm)  # wanted range is the same as norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]  # scale to a different range.
    # add other conditions here. For example, an error message

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.vector_field = Vector_Field()

        self.init_step_size = 1 / 480
        self.decay_rate = 1
        self.iterations = 3
        self.step_size = [(self.decay_rate ** i) * self.init_step_size for i in range(self.iterations)]
        print('step size: ', [s * 480 for s in self.step_size])

    def forward(self, x, pts):
        vector_field, dist_map = self.vector_field(x)

        dist_pred = dist_map
        dist_map = torch.clip(dist_map, 0, 0.3)
        dist_map = normalize(dist_map, newRange=(0, 25)) / 480

        eem_vector_field = vector_field[:, :2, ...]
        lumen_vector_field = vector_field[:, 2:, ...]

        eem_dist_map = dist_map[:, 0, :, :].unsqueeze(1)
        lumen_dist_map = dist_map[:, 1, :, :].unsqueeze(1)

        pts_iter = [pts.clone()]
        for iter in range(self.iterations):
            h = pts[:, :, :, 1]
            w = pts[:, :, :, 0]
            # map to [-1, 1]
            h_mapped = 2 * h - 1
            w_mapped = 2 * w - 1
            # clamp values to [-1, 1]
            h_mapped = torch.clamp(h_mapped, min=-1, max=1)
            w_mapped = torch.clamp(w_mapped, min=-1, max=1)

            eem_vector = F.grid_sample(eem_vector_field,
                                       torch.stack([w_mapped, h_mapped], dim=-1)[:, 0, :, :].unsqueeze(1),
                                       align_corners=True)  # in px units
            lumen_vector = F.grid_sample(lumen_vector_field,
                                         torch.stack([w_mapped, h_mapped], dim=-1)[:, 1, :, :].unsqueeze(1),
                                         align_corners=True)  # in px units

            eem_vector = torch.transpose(eem_vector.squeeze(2), 1, 2)
            lumen_vector = torch.transpose(lumen_vector.squeeze(2), 1, 2)

            eem_step = F.grid_sample(eem_dist_map,
                                       torch.stack([w_mapped, h_mapped], dim=-1)[:, 0, :, :].unsqueeze(1),
                                       align_corners=True)  # in px units
            lumen_step = F.grid_sample(lumen_dist_map,
                                         torch.stack([w_mapped, h_mapped], dim=-1)[:, 1, :, :].unsqueeze(1),
                                         align_corners=True)  # in px units

            eem_step = torch.transpose(eem_step.squeeze(2), 1, 2)
            lumen_step = torch.transpose(lumen_step.squeeze(2), 1, 2)

            eem_pts = pts[:, 0, :, :] + (eem_vector * eem_step)
            lumen_pts = pts[:, 1, :, :] + (lumen_vector * lumen_step)

            pts = torch.stack([eem_pts, lumen_pts], dim=1)

            pts_iter.append(pts.clone())

        vector_field_pred = torch.stack([eem_vector_field.permute(0, 2, 3, 1), lumen_vector_field.permute(0, 2, 3, 1)],
                                        dim=1)  #

        return pts, vector_field_pred, dist_pred

class GCN(nn.Module):
    def __init__(self, adj_mat, up_mat, graph_dim, num_levels, fpn_dim, edge_list):
        super().__init__()

        self.adj_mat = adj_mat
        self.up_mat = up_mat
        self.graph_dim = graph_dim
        self.edge_list = edge_list

        self.feat_dim = fpn_dim * num_levels
        self.step_size = 0.05

        self.block1 = Graph_Block(self.feat_dim + 2, self.graph_dim, self.graph_dim, adj_mat[0])
        self.block2 = Graph_Block(self.feat_dim + self.graph_dim + 2, self.graph_dim, self.graph_dim, adj_mat[1])
        self.block3 = Graph_Block(self.feat_dim + self.graph_dim + 2, self.graph_dim, self.graph_dim, adj_mat[2])
        self.block4 = Graph_Block(self.feat_dim + self.graph_dim + 2, self.graph_dim, self.graph_dim, adj_mat[3])

        self.upsample = Pool()

    def normal(self, edge_list, points):
        """
        :param edge_list: torch.Tensor(PTS, 2) look up table to see what nodes are connected to a given node
        :param points: torch.Tensor(BS, PTS, 2) point coordinates
        :return: normalised direction of normal for each point torch.Tensor(BS, PTS, 2).
        """
        # for each point, find points that connect to it.
        adj_nodes = points[:, edge_list]

        # find vector from point to each neighbour, and the normal to that vector.
        edge_vectors_1 = self.norm(adj_nodes[:, :, 0, :] - points)
        edge_normal_1 = self.norm(torch.stack([- edge_vectors_1[:, :, 1], edge_vectors_1[:, :, 0]], 2))
        edge_vectors_2 = self.norm(points - adj_nodes[:, :, 1, :])
        edge_normal_2 = self.norm(torch.stack([- edge_vectors_2[:, :, 1], edge_vectors_2[:, :, 0]], 2))

        # average normal vectors
        edge_normal = torch.stack([edge_normal_1, edge_normal_2], dim=2)
        node_normal = self.norm(edge_normal.mean(dim=2))

        return node_normal

    def norm(self, vec, dim=2):
        return vec / torch.linalg.norm(vec, dim=dim, keepdims=True)

    def forward(self, feat_maps, points):
        """
        inp:
        initial point coords Torch([BS, N, I, 2])
        feat_maps list of Torch([BS, I, C, H, W])
        """
        bs = points.shape[0]

        # BLOCK 1
        norm1 = self.normal(self.edge_list[0], points.reshape(bs, -1, 2))
        feat1 = self.sample_features(feat_maps, points)
        inp1 = torch.cat([feat1, points], -1)
        x1, mag1 = self.block1(inp1.reshape(bs, 16, -1))
        out1 = points.reshape(bs, -1, 2) + (self.step_size * norm1 * mag1)

        # BLOCK 2
        x2 = self.upsample(x1, self.up_mat[0])
        out2 = self.upsample(out1, self.up_mat[0])
        norm2 = self.normal(self.edge_list[1], out2)
        feat2 = self.sample_features(feat_maps, out2.reshape(bs, 2, -1, 2))
        inp2 = torch.cat([x2, feat2.reshape(bs, 32, -1), out2], -1)
        x2, mag2 = self.block2(inp2)
        out2 = out2.reshape(bs, -1, 2) + (self.step_size * norm2 * mag2)

        # BLOCK 3
        x3 = self.upsample(x2, self.up_mat[1])
        out3 = self.upsample(out2, self.up_mat[1])
        norm3 = self.normal(self.edge_list[2], out3)
        feat3 = self.sample_features(feat_maps, out3.reshape(bs, 2, -1, 2))
        inp3 = torch.cat([x3, feat3.reshape(bs, 64, -1), out3], -1)
        x3, mag3 = self.block3(inp3)
        out3 = out3.reshape(bs, -1, 2) + (self.step_size * norm3 * mag3)

        # BLOCK 4
        x4 = self.upsample(x3, self.up_mat[2])
        out4 = self.upsample(out3, self.up_mat[2])
        norm4 = self.normal(self.edge_list[3], out4)
        feat4 = self.sample_features(feat_maps, out4.reshape(bs, 2, -1, 2))
        inp4 = torch.cat([x4, feat4.reshape(bs, 128, -1), out4], -1)
        x4, mag4 = self.block4(inp4)
        out4 = out4.reshape(bs, -1, 2) + (self.step_size * norm4 * mag4)

        return [out1, out2, out3, out4]

    def sample_features(self, pooled_feat, points):
        """
        inp:
        points:  Tensor([BS, I, PTS, 2]) in range [0,1]
        feature maps: list Tensor([BS, I, C, H, W])

        out: Tensor([BS, I, PTS, F]) where F is the sum of all Cs
        """
        bs, I, pts, xy = points.shape
        #print([f.shape for f in pooled_feat], points.shape)
        sampled_feat = []
        for feat in pooled_feat:

            c = feat.shape[1]
            points = points.reshape(bs, I * pts, xy)

            #print('feat:', feat.shape)

            h = points[:, :, 1]
            w = points[:, :, 0]
            # map from [0, 1] to [-1, 1]
            h_mapped = 2 * h - 1
            w_mapped = 2 * w - 1
            # clamp values to [-1, 1]
            h_mapped = torch.clamp(h_mapped, min=-1, max=1)
            w_mapped = torch.clamp(w_mapped, min=-1, max=1)
            sampled = F.grid_sample(feat, torch.stack([w_mapped, h_mapped], dim=-1).unsqueeze(1),
                                    align_corners=True)
            sampled = sampled.reshape(bs, I, pts, c)
            sampled_feat.append(sampled)

        sampled_feat = torch.cat(sampled_feat, -1)
        return sampled_feat

class Graph_Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, adj):
        super().__init__()

        self.edge_index = adj._indices()

        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.norm2 = nn.InstanceNorm1d(out_dim)
        self.relu = nn.ReLU()

        self.residual = nn.Linear(in_dim, out_dim)

        self.gcn_out = GCNConv(out_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inp = x

        x = self.gcn1(x, self.edge_index)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.gcn2(x, self.edge_index)
        x = self.norm2(x)
        x = self.relu(x)

        x = x + self.residual(inp)

        out = self.sigmoid(self.gcn_out(x, self.edge_index)) - 0.5 # scale output to [-0.5, 0.5]

        return x, out


