"""
Model file for zsgnet
Author: Arka Sadhu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fpn_resnet import FPN_backbone, FPN_prune_backbone
from anchors import create_grid
import ssd_vgg
from typing import Dict, Any
from extended_config import cfg as conf
from dat_loader import get_data
from afs import AdaptiveFeatureSelection,PrunedFeatureSelection,LanguagePrune
from garan import GaranAttention
import numpy as np
import cv2
import time
from skipnet import *
from darknet import *

# conv2d, conv2d_relu are adapted from
# https://github.com/fastai/fastai/blob/5c4cefdeaf11fdbbdf876dbe37134c118dca03ad/fastai/layers.py#L98
def conv2d(ni: int, nf: int, ks: int = 3, stride: int = 1,
           padding: int = None, bias=False) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None:
        padding = ks//2
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                     padding=padding, bias=bias)


def conv2d_relu(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None,
                bn: bool = False, bias: bool = False) -> nn.Sequential:
    """
    Create a `conv2d` layer with `nn.ReLU` activation
    and optional(`bn`) `nn.BatchNorm2d`: `ni` input, `nf` out
    filters, `ks` kernel, `stride`:stride, `padding`:padding,
    `bn`: batch normalization.
    """
    layers = [conv2d(ni, nf, ks=ks, stride=stride,
                     padding=padding, bias=bias), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

def gesture_attention(attens,gts,sizes):
    RADIUS = 10
    new_attens = []
    #print(len(attens))
    for i,gt in enumerate(gts):
        gt_pointed = [(gt[0]+gt[2])/2,(gt[1]+gt[3])/2]
        for j,atten in enumerate(attens):

            atten_size = atten.shape
            #print(atten_size)
            atten[i] = atten[i]*0.5
            attens[j][i] = atten[i]
            img_size = sizes[i]
            pointed = [gt_pointed[0]/img_size[1]*atten_size[1],gt_pointed[1]/img_size[0]*atten_size[0]]
            pointed = [int(pointed[0]),int(pointed[1])]
            for m in range(pointed[0]-int(RADIUS/(j+1)),pointed[0]+int(RADIUS/(j+1))):
                if m >= atten_size[1] or m < 0:
                    continue
                for n in range(pointed[1]-int(RADIUS/(j+1)),pointed[1]+int(RADIUS/(j+1))):
                    if n >= atten_size[0] or n < 0:
                        continue
                    atten[i,n,m]=2*atten[i,n,m]
                    attens[j][i,n,m] = atten[i,n,m]

            #new_attens.append(attens)
    return attens

def gesture_attention_new(attens,pointed):
    pointed = (pointed +1)/2
    new_attens = []
    #print("Before: ",pointed[0])
    for i,atten in enumerate(attens):
        #atten = atten*0.5
        #mask = 0.5*np.ones(atten.shape)
        atten = 0.05*atten
        if i==0:
            pointed[:,0] = atten.shape[2]*pointed[:,0]
            pointed[:,1] = atten.shape[1]*pointed[:,1]
        else:
            pointed[:,0] = pointed[:,0]/2
            pointed[:,1] = pointed[:,1]/2
        for i in range(atten.shape[0]):
            atten[i,int(pointed[i,1]),int(pointed[i,0])] = 20*atten[i,int(pointed[i,1]),int(pointed[i,0])]
        #print("Pointed: ",pointed[0])
        #print("Mask: ",mask[0])
        #print("After: ",pointed[0])
        #cv2.imwrite("mask.jpg",255*mask[0,0:2])
        new_attens.append(atten)

    return new_attens
    

class BackBone(nn.Module):
    """
    A general purpose Backbone class.
    For a new network, need to redefine:
    --> encode_feats
    Optionally after_init
    """

    def __init__(self, encoder: nn.Module, cfg: dict, out_chs=256):
        """
        Make required forward hooks
        """
        super().__init__()
        self.device = torch.device(cfg.device)
        self.encoder = encoder
        self.cfg = cfg
        self.out_chs = out_chs
        self.after_init()

    def after_init(self):
        pass

    def num_channels(self):
        raise NotImplementedError

    def concat_we(self, x, we, only_we=False, only_grid=False):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W (one feature map)
        we: B x wdim (the language vector)
        Output: concatenated word embedding and grid centers
        """
        # Both cannot be true
        assert not (only_we and only_grid)

        # Create the grid
        grid = create_grid((x.size(2), x.size(3)),
                           flatten=False).to(self.device)
        grid = grid.permute(2, 0, 1).contiguous()

        # TODO: Slightly cleaner implementation?
        grid_tile = grid.view(
            1, grid.size(0), grid.size(1), grid.size(2)).expand(
            we.size(0), grid.size(0), grid.size(1), grid.size(2))

        # In case we only need the grid
        # Basically, don't use any image/language information
        if only_grid:
            return grid_tile

        # Expand word embeddings
        word_emb_tile = we.view(
            we.size(0), we.size(1), 1, 1).expand(
                we.size(0), we.size(1), x.size(2), x.size(3))

        # In case performing image blind (requiring only language)
        if only_we:
            return word_emb_tile

        # Concatenate along the channel dimension
        return torch.cat((x, word_emb_tile, grid_tile), dim=1)

    def encode_feats(self, inp):
        return self.encoder(inp)

    def forward(self, inp, we=None,
                only_we=False, only_grid=False):
        """
        expecting word embedding of shape B x WE.
        If only image features are needed, don't
        provide any word embedding
        """
        feats,att_maps,gest = self.encode_feats(inp,we)
        # If we want to do normalization of the features
        if self.cfg['do_norm']:
            feats = [
                feat / feat.norm(dim=1).unsqueeze(1).expand(*feat.shape)
                for feat in feats
            ]

        # For language blind setting, can directly return the features
        if we is None:
            return feats

        if self.cfg['do_norm']:
            b, wdim = we.shape
            we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)

        out = [self.concat_we(
            f, we, only_we=only_we, only_grid=only_grid) for f in feats]

        return out,att_maps,gest


class SkipNetBackBone(BackBone):
    def after_init(self):
        self.inplanes = 64
        block = Bottleneck
        layers = [3, 8, 36, 3]
        embed_dim = 10
        num_classes=1000
        hidden_dim = 10
        gate_type='rnn'

        self.num_layers = layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # going to have 4 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.
        self._make_group(block, 64, layers[0], group_id=1, pool_size=56)
        self._make_group(block, 128, layers[1], group_id=2, pool_size=28)
        self._make_group(block, 256, layers[2], group_id=3, pool_size=14)
        self._make_group(block, 512, layers[3], group_id=4, pool_size=7)

        self.control = RNNGatePolicy(embed_dim, hidden_dim, rnn_type='lstm')

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.softmax = nn.Softmax()

        # save everything
        self.saved_actions = {}
        self.saved_dists = {}
        self.saved_outputs = {}
        self.saved_targets = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
	
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=AdaptiveFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=AdaptiveFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=AdaptiveFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)

        self.gesture_regression1 = nn.Linear(64*56*56,1024)
        self.gesture_regression2 = nn.Linear(1024,128)
        self.gesture_regression3 = nn.Linear(128,2)

    def _make_group(self, block, planes, layers, group_id=1, pool_size=56):
        """ Create the whole group """
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def repackage_hidden(self):
        self.control.hidden = repackage_hidden(self.control.hidden)

    def _make_layer_v2(self, block, planes, stride=1, pool_size=56):
        """ create one block and optional a gate module """
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))
        return downsample, layer, gate_layer

    def num_channels(self):
         return [512,1024,2048]

    def encode_feats(self, x,lang):

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
	
        x_gest  = x
        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        actions = []
        dists = []

        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer
        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob, action, dist = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        current_device = torch.cuda.current_device()
        actions.append(action)
        dists.append(dist)
        x1 = None
        x2 = None
        x3 = None
        x4 = None
        for g in range(4):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x)*x + (1-mask).expand_as(prev)*prev
                if not (g == 3 and (i == self.num_layers[g] - 1)):
                    gate_feature = getattr(self,
                                           'group{}_gate{}'.format(g+1, i))(x)
                    mask, gprob, action, dist = self.control(gate_feature)
                    gprobs.append(gprob)
                    masks.append(mask.squeeze())
                    actions.append(action)
                    dists.append(dist)
            if g==0:
                x1 = x
            if g==1:
                x2 = x
            if g==2:
                x3 = x
            if g==3:
                x4 = x

        gest = torch.reshape(x_gest,(-1,64*56*56))
        gest = F.leaky_relu(self.gesture_regression1(gest))
        gest = F.leaky_relu(self.gesture_regression2(gest))
        gest = self.gesture_regression3(gest)
        # print(lang.size())
        x2_ = self.afs_stage0(lang,[x2, x3, x4])
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = self.afs_stage1(lang,[x2, x3, x4])
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = self.afs_stage2(lang,[x2, x3, x4])
        x4_,E_3 = self.garan_stage2(lang, x4_)
        feats = self.fpn([x2_, x3_, x4_])
        return feats,[E_1,E_2,E_3],gest


class DarkNetBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=AdaptiveFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=AdaptiveFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=AdaptiveFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.gesture_regression1 = nn.Linear(1024*7*7,1024)
        self.gesture_regression2 = nn.Linear(1024,128)
        self.gesture_regression3 = nn.Linear(128,2)
    def num_channels(self):
        #return [self.encoder.layer2[-1].conv3.out_channels,
        #        self.encoder.layer3[-1].conv3.out_channels,
        #        self.encoder.layer4[-1].conv3.out_channels]
        return [256,512,1024]

    def encode_feats(self, inp,lang):
        x = self.encoder(inp)
        #print(x[0].shape,x[1].shape,x[2].shape)
        #x = self.encoder.conv1(inp)
        #x1 = self.encoder.maxpool(x)
        x2 = x[2]
        x3 = x[1]
        x4 = x[0]
        
        gest = torch.reshape(x4,(-1,1024*7*7))
        gest = F.leaky_relu(self.gesture_regression1(gest))
        gest = F.leaky_relu(self.gesture_regression2(gest))
        gest = self.gesture_regression3(gest)
        # print(lang.size())
        x2_ = self.afs_stage0(lang,[x2, x3, x4])
        #print(x2_.shape)
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = self.afs_stage1(lang,[x2, x3, x4])
        #print(x3_.shape)
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = self.afs_stage2(lang,[x2, x3, x4])
        #print(x4_.shape)
        x4_,E_3 = self.garan_stage2(lang, x4_)
        feats = self.fpn([x2_, x3_, x4_])
        return feats,[E_1,E_2,E_3],gest


class SuffleNetBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=AdaptiveFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=AdaptiveFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=AdaptiveFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.gesture_regression1 = nn.Linear(464*7*7,1024)
        self.gesture_regression2 = nn.Linear(1024,128)
        self.gesture_regression3 = nn.Linear(128,2)
    def num_channels(self):
        #return [self.encoder.layer2[-1].conv3.out_channels,
        #        self.encoder.layer3[-1].conv3.out_channels,
        #        self.encoder.layer4[-1].conv3.out_channels]
        return [116,232,464]

    def encode_feats(self, inp,lang):
        x = self.encoder.conv1(inp)
        x1 = self.encoder.maxpool(x)
        x2 = self.encoder.stage2(x1)
        #print("x2: ",x2.shape )
        x3 = self.encoder.stage3(x2)
        #print("x3: ",x3.shape)
        x4 = self.encoder.stage4(x3)
        #print("x4: ",x4.shape)
        gest = torch.reshape(x4,(-1,464*7*7))
        gest = F.leaky_relu(self.gesture_regression1(gest))
        gest = F.leaky_relu(self.gesture_regression2(gest))
        gest = self.gesture_regression3(gest)
        # print(lang.size())
        x2_ = self.afs_stage0(lang,[x2, x3, x4])
        #print(x2_.shape)
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = self.afs_stage1(lang,[x2, x3, x4])
        #print(x3_.shape)
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = self.afs_stage2(lang,[x2, x3, x4])
        #print(x4_.shape)
        x4_,E_3 = self.garan_stage2(lang, x4_)
        feats = self.fpn([x2_, x3_, x4_])
        return feats,[E_1,E_2,E_3],gest


class ShufflePruneNetBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_prune_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=PrunedFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=PrunedFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=PrunedFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)
        self.pruner = LanguagePrune(256,3).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)
        #self.gesture_regression1 = nn.Linear(464*7*7,1024)
        self.gesture_regression1 = nn.Linear(512*7*7,1024)

        self.gesture_regression2 = nn.Linear(1024,128)
        self.gesture_regression3 = nn.Linear(128,2)

        self.vis_stage1 = nn.Conv2d(116,512,1).to(self.device)
        self.vis_stage2 = nn.Conv2d(232,512,1).to(self.device)
        self.vis_stage3 = nn.Conv2d(464,512,1).to(self.device)
        
        self.pooler2 = nn.Conv2d(24,116,1, stride=2)
        self.pooler3 = nn.Conv2d(116,232,1, stride=2)
        self.pooler4 = nn.Conv2d(232,464,1, stride=2)

    def num_channels(self):
        #return [self.encoder.layer2[-1].conv3.out_channels,
        #        self.encoder.layer3[-1].conv3.out_channels,
        #        self.encoder.layer4[-1].conv3.out_channels]
        #return [116,232,464]
        return [512,512,512]

    def encode_feats(self, inp,lang):

        prune_weights = self.pruner(lang)
        w2 = prune_weights[:,:,0].view(-1, 1, 1, 1).expand(-1,512,28,28)
        w3 = prune_weights[:,:,1].view(-1, 1, 1, 1).expand(-1,512,14,14)
        w4 = prune_weights[:,:,2].view(-1, 1, 1, 1).expand(-1,512,7,7)
        x = self.encoder.conv1(inp)
        x1 = self.encoder.maxpool(x)
        #print("x1: ",x1.shape)
        x2 = self.encoder.stage2(x1)
        #print("x2: ",self.pooler1(x1).shape )
        x3 = w3[:,0:232,:,:]*self.encoder.stage3(x2) + self.pooler3(x2)
        
        #print("x3: ",x3.shape)
        x4 = w4[:,0:464,:,:]*self.encoder.stage4(x3) + self.pooler4(x3)

        x2 = self.vis_stage1(x2)
        x3 = self.vis_stage2(x3)
        x4 = self.vis_stage3(x4)
        #print("x4: ",x4.shape)
        gest = torch.reshape(x4,(-1,512*7*7))
        gest = F.leaky_relu(self.gesture_regression1(gest))
        gest = F.leaky_relu(self.gesture_regression2(gest))
        gest = self.gesture_regression3(gest)
        # print(lang.size())

        x2_ = w2*self.afs_stage0(lang,[x2, x3, x4]) + x2#,prune_weights)
        
        
        
        #print("x2_",x2_.shape,prune_weights.shape)
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = w3*self.afs_stage1(lang,[x2, x3, x4]) + x3#,prune_weights)
        #print("x3_",x3_.shape,x3.shape)
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = w4*self.afs_stage2(lang,[x2, x3, x4]) + x4#,prune_weights)
        #print("x4_",x4_.shape,x4.shape)
        x4_,E_3 = self.garan_stage2(lang, x4_)
        #print("Final: ",x2_.shape,x3_.shape,x4_.shape)
        feats = self.fpn([x2_, x3_, x4_],[w2,w3,w4])
        
        #print("Final: ",x2_.shape,x3_.shape,x4_.shape)
        #print("Final2: ",E_1.shape,E_2.shape,E_3.shape)
        return feats,[E_1,E_2,E_3],gest


class RetinaBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=AdaptiveFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=AdaptiveFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=AdaptiveFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.gesture_regression1 = nn.Linear(64*56*56,1024)
        self.gesture_regression2 = nn.Linear(1024,128)
        self.gesture_regression3 = nn.Linear(128,2)
    def num_channels(self):
        return [self.encoder.layer2[-1].conv3.out_channels,
                self.encoder.layer3[-1].conv3.out_channels,
                self.encoder.layer4[-1].conv3.out_channels]

    def encode_feats(self, inp,lang):
        x = self.encoder.conv1(inp)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        gest = torch.reshape(x,(-1,64*56*56))
        gest = F.leaky_relu(self.gesture_regression1(gest))
        gest = F.leaky_relu(self.gesture_regression2(gest))
        gest = self.gesture_regression3(gest)
        # print(lang.size())
        x2_ = self.afs_stage0(lang,[x2, x3, x4])
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = self.afs_stage1(lang,[x2, x3, x4])
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = self.afs_stage2(lang,[x2, x3, x4])
        x4_,E_3 = self.garan_stage2(lang, x4_)
        feats = self.fpn([x2_, x3_, x4_])
        return feats,[E_1,E_2,E_3],gest

class RetinaPruneBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=PrunedFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=PrunedFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=AdaptiveFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)
        self.pruner = LanguagePrune(256,3).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.gesture_regression1 = nn.Linear(64*56*56,1024)
        self.gesture_regression2 = nn.Linear(1024,128)
        self.gesture_regression3 = nn.Linear(128,2)

        #self.vis_stage1 = nn.Conv2d(116,512,1).to(self.device)
        self.vis_stage2 = nn.Conv2d(1024,512,1).to(self.device)
        self.vis_stage3 = nn.Conv2d(2048,512,1).to(self.device)
        
        #self.pooler2 = nn.Conv2d(24,116,1, stride=2)
        self.pooler3 = nn.Conv2d(512,1024,1, stride=2)
        self.pooler4 = nn.Conv2d(1024,2048,1, stride=2)
    def num_channels(self):
        return [self.encoder.layer2[-1].conv3.out_channels,
                self.encoder.layer3[-1].conv3.out_channels,
                self.encoder.layer4[-1].conv3.out_channels]

    def encode_feats(self, inp,lang):

        prune_weights = self.pruner(lang)
        w2 = prune_weights[:,:,0].view(-1, 1, 1, 1).expand(-1,2048,28,28)
        w3 = prune_weights[:,:,1].view(-1, 1, 1, 1).expand(-1,2048,14,14)
        w4 = prune_weights[:,:,2].view(-1, 1, 1, 1).expand(-1,2048,7,7)
        
        x = self.encoder.conv1(inp)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = w3[:,0:1024,:,:]*self.encoder.layer3(x2) + self.pooler3(x2)
        x4 = w4[:,0:2048,:,:]*self.encoder.layer4(x3) + self.pooler4(x3)
        #print(x2.shape,x3.shape,x4.shape)
        
        #_x2 = self.vis_stage1(x2)
        _x3 = self.vis_stage2(x3)
        _x4 = self.vis_stage3(x4)
        
        gest = torch.reshape(x,(-1,64*56*56))
        gest = F.leaky_relu(self.gesture_regression1(gest))
        gest = F.leaky_relu(self.gesture_regression2(gest))
        gest = self.gesture_regression3(gest)
        # print(lang.size())
        x2_ = w2[:,0:512,:]*self.afs_stage0(lang,[x2, x3, x4]) + x2
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = w3[:,0:512,:]*self.afs_stage1(lang,[x2, x3, x4]) + _x3
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = w4[:,0:512,:]*self.afs_stage2(lang,[x2, x3, x4]) + _x4
        x4_,E_3 = self.garan_stage2(lang, x4_)
        feats = self.fpn([x2_, x3_, x4_])
        return feats,[E_1,E_2,E_3],gest


class MobilenetBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone([512,512,512], self.cfg, feat_size=self.out_chs).to(self.device)
        self.afs_stage0=AdaptiveFeatureSelection(0,[],2,list(self.num_chs[1:]),self.num_chs[0],256,256,512).to(self.device)
        self.afs_stage1=AdaptiveFeatureSelection(1,[self.num_chs[0]],1,[self.num_chs[-1]],self.num_chs[1],256,256,512).to(self.device)
        self.afs_stage2=AdaptiveFeatureSelection(2,list(self.num_chs[:-1]),0,[],self.num_chs[-1],256,256,512).to(self.device)

        self.garan_stage0 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage1 = GaranAttention(256, 512, n_head=2).to(self.device)
        self.garan_stage2 = GaranAttention(256, 512, n_head=2).to(self.device)
    def num_channels(self):
        #return [24,24,24] 
        return [16,24,32]
        #return [32,64,96]
        #return [self.encoder.features[5].conv[-1].out_channels,self.encoder.features[5].conv[-1].out_channels,self.encoder.features[5].conv[-1].out_channels]
        #return [self.encoder.layer2[-1].conv3.out_channels,
        #        self.encoder.layer2[-1].conv3.out_channels,
        #        self.encoder.layer2[-1].conv3.out_channels]
    def encode_feats(self, inp,lang):
        #print(encoder)
        #x = self.encoder.conv1(inp)
        #x = self.encoder.bn1(x)
        #x = self.encoder.relu(x)
        #x = self.encoder.maxpool(x)
        #x1 = self.encoder.layer1(x)
        #x2 = self.encoder.layer2(x1)
        #x3 = self.encoder.layer3(x2)
        #x4 = self.encoder.layer4(x3)


        ##Working one
        x1  = self.encoder.features[0](inp)
        x2  = self.encoder.features[1](x1)
        x3  = self.encoder.features[2](x2)
        x3  = self.encoder.features[3](x3)
        x4  = self.encoder.features[4](x3)
        ## Working end

        #x1  = self.encoder.features[0](inp)
        #x1  = self.encoder.features[1](x1)
        #x1  = self.encoder.features[2](x1)
        ###Test
        #x1  = self.encoder.features[3](x1)
        #x2  = self.encoder.features[4](x1)
        #x3  = self.encoder.features[5](x2)
        #x3  = self.encoder.features[6](x3)
        #x3  = self.encoder.features[7](x3)
        #x4  = self.encoder.features[8](x3)
        #x4  = self.encoder.features[9](x4)
        #x4  = self.encoder.features[10](x4)
        #x4  = self.encoder.features[11](x4)
        ##x4 = self.encoder.features[16](x4)
        #x4 = self.encoder.features[17](x4)
        #x1 = x
        #x2 = x
        #x3 = x2
        #x4 = x2
        #print("Dulanga: ",lang.size(),x2.size(),x3.size(),x4.size())
        x2_ = self.afs_stage0(lang,[x2, x3, x4])
        x2_,E_1=self.garan_stage0(lang,x2_)
        x3_ = self.afs_stage1(lang,[x2, x3, x4])
        x3_,E_2 = self.garan_stage1(lang, x3_)
        x4_ = self.afs_stage2(lang,[x2, x3, x4])
        x4_,E_3 = self.garan_stage2(lang, x4_)
        feats = self.fpn([x2_, x3_, x4_])
        #feats = self.fpn([x2_, x3_,x2_])
        return feats,[E_1,E_2,E_3]

class SSDBackBone(BackBone):
    """
    ssd_vgg.py already implements encoder
    """

    def encode_feats(self, inp):
        return self.encoder(inp)


class ZSGNet(nn.Module):
    """
    The main model
    Uses SSD like architecture but for Lang+Vision
    """

    def __init__(self, backbone, n_anchors=1, final_bias=0., cfg=None):
        super().__init__()
        # assert isinstance(backbone, BackBone)
        self.backbone = backbone

        # Assume the output from each
        # component of backbone will have 256 channels
        self.device = torch.device(cfg.device)

        self.cfg = cfg

        # should be len(ratios) * len(scales)
        self.n_anchors = n_anchors

        self.emb_dim = cfg['emb_dim']
        self.bid = cfg['use_bidirectional']
        self.lstm_dim = cfg['lstm_dim']

        # Calculate output dimension of LSTM
        self.lstm_out_dim = self.lstm_dim * (self.bid + 1)

        # Separate cases for language, image blind settings
        if self.cfg['use_lang'] and self.cfg['use_img']:
            self.start_dim_head = self.lstm_dim*(self.bid+1) + 256 + 2
        elif self.cfg['use_img'] and not self.cfg['use_lang']:
            # language blind
            self.start_dim_head = 256
        elif self.cfg['use_lang'] and not self.cfg['use_img']:
            # image blind
            self.start_dim_head = self.lstm_dim*(self.bid+1)
        else:
            # both image, lang blind
            self.start_dim_head = 2

        # If shared heads for classification, box regression
        # This is the config used in the paper
        if self.cfg['use_same_atb']:
            bias = torch.zeros(5 * self.n_anchors)
            bias[torch.arange(4, 5 * self.n_anchors, 5)] = -4
            self.att_reg_box = self._head_subnet(
                5, self.n_anchors, final_bias=bias,
                start_dim_head=self.start_dim_head
            )
        # This is not used. Kept for historical purposes
        else:
            self.att_box = self._head_subnet(
                1, self.n_anchors, -4., start_dim_head=self.start_dim_head)
            self.reg_box = self._head_subnet(
                4, self.n_anchors, start_dim_head=self.start_dim_head)

        self.lstm = nn.LSTM(self.emb_dim, self.lstm_dim,
                            bidirectional=self.bid, batch_first=False)
        self.after_init()

    def after_init(self):
        "Placeholder if any child class needs something more"
        pass

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256,
                     start_dim_head=256):
        """
        Convenience function to create attention and regression heads
        """
        layers = [conv2d_relu(start_dim_head, chs, bias=True)]
        layers += [conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        return nn.Sequential(*layers)

    def permute_correctly(self, inp, outc):
        """
        Basically square box features are flattened
        """
        # inp is features
        # B x C x H x W -> B x H x W x C
        out = inp.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, outc)
        return out

    def concat_we(self, x, we, append_grid_centers=True):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W
        we: B x wdim
        """
        b, wdim = we.shape
        we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)
        word_emb_tile = we.view(we.size(0), we.size(1),
                                1, 1).expand(we.size(0),
                                             we.size(1),
                                             x.size(2), x.size(3))

        if append_grid_centers:
            grid = create_grid((x.size(2), x.size(3)),
                               flatten=False).to(self.device)
            grid = grid.permute(2, 0, 1).contiguous()
            grid_tile = grid.view(1, grid.size(0), grid.size(1), grid.size(2)).expand(
                we.size(0), grid.size(0), grid.size(1), grid.size(2))

            return torch.cat((x, word_emb_tile, grid_tile), dim=1)
        return torch.cat((x, word_emb_tile), dim=1)

    def lstm_init_hidden(self, bs):
        """
        Initialize the very first hidden state of LSTM
        Basically, the LSTM should be independent of this
        """
        if not self.bid:
            hidden_a = torch.randn(1, bs, self.lstm_dim)
            hidden_b = torch.randn(1, bs, self.lstm_dim)
        else:
            hidden_a = torch.randn(2, bs, self.lstm_dim)
            hidden_b = torch.randn(2, bs, self.lstm_dim)

        hidden_a = hidden_a.to(self.device)
        hidden_b = hidden_b.to(self.device)

        return (hidden_a, hidden_b)

    def apply_lstm(self, word_embs, qlens, max_qlen, get_full_seq=False):
        """
        Applies lstm function.
        word_embs: word embeddings, B x seq_len x 300
        qlen: length of the phrases
        Try not to fiddle with this function.
        IT JUST WORKS
        """
        # B x T x E
        bs, max_seq_len, emb_dim = word_embs.shape
        # bid x B x L
        self.hidden = self.lstm_init_hidden(bs)
        # B x 1, B x 1
        qlens1, perm_idx = qlens.sort(0, descending=True)
        # B x T x E (permuted)
        qtoks = word_embs[perm_idx]
        # T x B x E
        embeds = qtoks.permute(1, 0, 2).contiguous()
        # Packed Embeddings
        # Code added by Zaland
        qlens1 = qlens1.cpu()

        # Zaland Code Ended
        packed_embed_inp = pack_padded_sequence(
            embeds, lengths=qlens1, batch_first=False)
        # To ensure no pains with DataParallel
        # self.lstm.flatten_parameters()
        lstm_out1, (self.hidden, _) = self.lstm(packed_embed_inp, self.hidden)

        # T x B x L
        lstm_out, req_lens = pad_packed_sequence(
            lstm_out1, batch_first=False, total_length=max_qlen)

        # TODO: Simplify getting the last vector
        masks = (qlens1-1).view(1, -1, 1).expand(max_qlen,
                                                 lstm_out.size(1), lstm_out.size(2))
        print(f"The device for masks is {masks.device}")
        print(f"The device for lstm_out is {lstm_out.device}")
        qvec_sorted = lstm_out.gather(0, masks.long())[0]

        qvec_out = word_embs.new_zeros(qvec_sorted.shape)
        qvec_out[perm_idx] = qvec_sorted
        # if full sequence is needed for future work
        if get_full_seq:
            lstm_out_1 = lstm_out.transpose(1, 0).contiguous()
            return lstm_out_1
        return qvec_out.contiguous()

    def forward(self, inp: Dict[str, Any]):
        """
        Forward method of the model
        inp0 : image to be used
        inp1 : word embeddings, B x seq_len x 300
        qlens: length of phrases

        The following is performed:
        1. Get final hidden state features of lstm
        2. Get image feature maps
        3. Concatenate the two, specifically, copy lang features
        and append it to all the image feature maps, also append the
        grid centers.
        4. Use the classification, regression head on this concatenated features
        The matching with groundtruth is done in loss function and evaluation
        """
        inp0 = inp['img']
        inp1 = inp['qvec']
        qlens = inp['qlens']
        max_qlen = int(qlens.max().item())
        req_embs = inp1[:, :max_qlen, :].contiguous()
        # Added by Zaland
        print(qlens.device)
        print(req_embs.device)
        # qlens = qlens.cpu()
        # req_embs = req_embs.cpu()
        # Finished Adding 
        req_emb = self.apply_lstm(req_embs, qlens, max_qlen)

        # image blind
        if self.cfg['use_lang'] and not self.cfg['use_img']:
            # feat_out = self.backbone(inp0)
            feat_out,E_attns = self.backbone(inp0, req_emb, only_we=True)

        # language blind
        elif self.cfg['use_img'] and not self.cfg['use_lang']:
            feat_out,E_attns = self.backbone(inp0)

        elif not self.cfg['use_img'] and not self.cfg['use_lang']:
            feat_out,E_attns = self.backbone(inp0, req_emb, only_grid=True)
        # see full language + image (happens by default)
        else:
            feat_out,E_attns,gest = self.backbone(inp0, req_emb)
        #E_attns = gesture_attention(E_attns,inp["orig_annot"],inp["img_size"])
        #E_attns = gesture_attention_new(E_attns,inp["gesture"])
        # Strategy depending on shared head or not
        if self.cfg['use_same_atb']:
            att_bbx_out = torch.cat([self.permute_correctly(
                self.att_reg_box(feature), 5) for feature in feat_out], dim=1)
            att_out = att_bbx_out[..., [-1]]
            bbx_out = att_bbx_out[..., :-1]
            #print(bbx_out[0].shape)
        else:
            att_out = torch.cat(
                [self.permute_correctly(self.att_box(feature), 1)
                 for feature in feat_out], dim=1)
            bbx_out = torch.cat(
                [self.permute_correctly(self.reg_box(feature), 4)
                 for feature in feat_out], dim=1)

        feat_sizes = torch.tensor([[f.size(2), f.size(3)]
                                   for f in feat_out]).to(self.device)

        # Used mainly due to dataparallel consistency
        num_f_out = torch.tensor([len(feat_out)]).to(self.device)

        out_dict = {}
        out_dict['att_out'] = att_out
        out_dict['bbx_out'] = bbx_out
        out_dict['feat_sizes'] = feat_sizes
        out_dict['num_f_out'] = num_f_out
        out_dict['att_maps'] = E_attns
        out_dict['gest'] = gest
        return out_dict


def get_default_net(num_anchors=1, cfg=None):
    """
    Constructs the network based on the config
    """
    if cfg['mdl_to_use'] == 'retina':
        #encoder = tvm.resnet50(True)
        encoder = tvm.resnet152(True)
        backbone = RetinaBackBone(encoder, cfg)
    if cfg['mdl_to_use'] == 'retinaprune':
        #encoder = tvm.resnet50(True)
        encoder = tvm.resnet152(True)
        backbone = RetinaPruneBackBone(encoder, cfg)
    elif cfg['mdl_to_use'] == 'skipnet':
        #encoder = tvm.resnet50(True)
        encoder = None
        backbone = SkipNetBackBone(encoder, cfg)
    elif cfg['mdl_to_use'] == 'shufflenet':
        #encoder = tvm.ShuffleNetV2(True,stages_repeat=[2,2,2],stages_out_channels=[512,1024,2048])
        encoder = tvm.shufflenet_v2_x1_0(True)
        #print(encoder)
        backbone = SuffleNetBackBone(encoder,cfg)
    elif cfg['mdl_to_use'] == 'darknet':
        #encoder = tvm.ShuffleNetV2(True,stages_repeat=[2,2,2],stages_out_channels=[512,1024,2048])
        encoder = Darknet(config_path='/home/Real-time-Global-Inference-Network-master/code_adapt/yolov3.cfg')
        encoder.load_weights('/home/Real-time-Global-Inference-Network-master/code_adapt/yolov3.weights')
        ## Text model
        #print(encoder)
        backbone = DarkNetBackBone(encoder,cfg)
    elif cfg['mdl_to_use'] == 'shuffleprune':
        #encoder = tvm.ShuffleNetV2(True,stages_repeat=[2,2,2],stages_out_channels=[512,1024,2048])
        encoder = tvm.shufflenet_v2_x1_0(True)
        #print(encoder)
        print("Using Prunable Network")
        backbone = ShufflePruneNetBackBone(encoder,cfg)
    elif cfg['mdl_to_use'] == 'mobilenet':
        #encoder = tvm.resnet50(True)
        #encoder = tvm.wide_resnet50_2(True)
        encoder = tvm.mobilenet_v2(True)
        print(encoder)
        #backbone = RetinaBackBone(encoder, cfg)
        print("Dulanga: Using Mobilenet Backend")
        backbone = MobilenetBackBone(encoder, cfg)
    elif cfg['mdl_to_use'] == 'ssd_vgg':
        encoder = ssd_vgg.build_ssd('train', cfg=cfg)
        encoder.vgg.load_state_dict(
            torch.load('./weights/vgg16_reducedfc.pth'))
        print('loaded pretrained vgg backbone')
        backbone = SSDBackBone(encoder, cfg)
        # backbone = encoder

    zsg_net = ZSGNet(backbone, num_anchors, cfg=cfg)
    print(zsg_net)
    return zsg_net


if __name__ == '__main__':
    # torch.manual_seed(0)
    cfg = conf
    cfg.mdl_to_use = 'ssd_vgg'
    cfg.ds_to_use = 'refclef'
    cfg.num_gpus = 1
    # cfg.device = 'cpu'
    device = torch.device(cfg.device)
    data = get_data(cfg)

    zsg_net = get_default_net(num_anchors=9, cfg=cfg)
    zsg_net.to(device)

    batch = next(iter(data.train_dl))
    for k in batch:
        batch[k] = batch[k].to(device)
    out = zsg_net(batch)
