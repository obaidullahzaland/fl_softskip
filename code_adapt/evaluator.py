import torch
from torch import nn
from anchors import (create_anchors, reg_params_to_bbox,
                     IoU_values, x1y1x2y2_to_y1x1y2x2)
from typing import Dict
import cv2
import numpy as np
from functools import partial
# from utils import reduce_dict


def reshape(box, new_size):
    """
    box: (N, 4) in y1x1y2x2 format
    new_size: (N, 2) stack of (h, w)
    """
    box[:, :2] = new_size * box[:, :2]
    box[:, 2:] = new_size * box[:, 2:]
    return box


class Evaluator(nn.Module):
    """
    To get the accuracy. Operates at training time.
    """

    def __init__(self, ratios, scales, cfg):
        super().__init__()
        self.cfg = cfg

        self.ratios = ratios
        self.scales = scales

        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.use_focal = cfg['use_focal']
        self.use_softmax = cfg['use_softmax']
        self.use_multi = cfg['use_multi']

        self.lamb_reg = cfg['lamb_reg']

        self.met_keys = ['Acc', 'MaxPos']
        self.anchs = None
        self.get_anchors = partial(
            create_anchors, ratios=self.ratios,
            scales=self.scales, flatten=True)

        self.acc = 0.0
        self.len = 0.0
        self.batch_acc = []
        self.acc_iou_threshold = self.cfg['acc_iou_threshold']

    def forward(self, out: Dict[str, torch.tensor],
                inp: Dict[str, torch.tensor],debug=False) -> Dict[str, torch.tensor]:

        annot = inp['annot']
        att_box = out['att_out']
        reg_box = out['bbx_out']
        feat_sizes = out['feat_sizes']
        num_f_out = out['num_f_out']
        #print(annot)
        #print(reg_box)
        device = att_box.device

        if len(num_f_out) > 1:
            num_f_out = int(num_f_out[0].item())
        else:
            num_f_out = int(num_f_out.item())

        feat_sizes = feat_sizes[:num_f_out, :]

        if self.anchs is None:
            feat_sizes = feat_sizes[:num_f_out, :]
            anchs = self.get_anchors(feat_sizes)
            anchs = anchs.to(device)
            self.anchs = anchs
        else:
            anchs = self.anchs
        att_box_sigmoid = torch.sigmoid(att_box).squeeze(-1)
        att_box_best, att_box_best_ids = att_box_sigmoid.max(1)
        #att_box_best, att_box_best_ids = torch.topk(att_box_sigmoid,10,dim=1)

        # self.att_box_best = att_box_best
        #att_box_best = att_box_best[:,9]
        #att_box_best_ids = att_box_best_ids[:,9]
        ious1 = IoU_values(annot, anchs)
        gt_mask, expected_best_ids = ious1.max(1)

        actual_bbox = reg_params_to_bbox(
            anchs, reg_box)
        #print(actual_bbox)
        best_possible_result, _,_ = self.get_eval_result(
            actual_bbox, annot, expected_best_ids)

        msk = None
        actual_result, pred_boxes,ious = self.get_eval_result(
            actual_bbox, annot, att_box_best_ids, msk)

        out_dict = {}
        out_dict['Acc'] = actual_result
        out_dict['MaxPos'] = best_possible_result
        out_dict['idxs'] = inp['idxs']

        reshaped_boxes = x1y1x2y2_to_y1x1y2x2(reshape(
            (pred_boxes + 1)/2, (inp['img_size'])))
        gt_boxes = x1y1x2y2_to_y1x1y2x2(reshape(
            (annot + 1)/2, (inp['img_size'])))
        #gest_ = (out['gest']+1)/2
        gest_in = (inp['gesture']+1)/2
        #print("Before: ",gest_)
        #print("Img Sizes: ",inp['img_size'])
        gest_in[:,0] = inp['img_size'][:,1]*gest_in[:,0]
        gest_in[:,1] = inp['img_size'][:,0]*gest_in[:,1]


        gest_out = (out['gest']+1)/2
        #print("Before: ",gest_)
        #print("Img Sizes: ",inp['img_size'])
        gest_out[:,0] = inp['img_size'][:,1]*gest_out[:,0]
        gest_out[:,1] = inp['img_size'][:,0]*gest_out[:,1]


        #print("Modified: ",gest_)

        #if (debug):
        #    acc_ = 0
        #    len_ = 0
        #    for im,size_,gt_box,pred_box,id_,result,ges_in,ges_out in zip(inp["img"],inp['img_size'],gt_boxes,reshaped_boxes,inp['idxs'],ious,gest_in,gest_out):
        #        im = im*255
        #        im = im.cpu().numpy()
        #        im = np.moveaxis(im, 0, -1)  
        #        size_ = size_.cpu().numpy()
        #        im = cv2.resize(im,dsize=(size_[1],size_[0]))
        #        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        #        im = cv2.rectangle(im, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0,0,0), 2)
        #        im = cv2.rectangle(im, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0,255,0), 2)
        #        #im = cv2.putText(im,str(result), (int(pred_box[0])+50, int(pred_box[1])-50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #        #im = cv2.circle(im,(int(ges_in[0]),int(ges_in[1])), 20, (0,0,255), -1)
        #        #im = cv2.circle(im,(int(ges_out[0]),int(ges_out[1])), 20, (0,0,255), -1)
        #        cv2.imwrite("results/test"+str(id_)+".jpg",im)
        #        loc = ((pred_box[0]+pred_box[2])/2,(pred_box[1]+pred_box[3])/2)
        #        if loc[0]>gt_box[0] and loc[0]<gt_box[2] and loc[1]>gt_box[1] and loc[1]<gt_box[3]:
        #            #self.acc+=1
        #            acc_+=1

        #        #self.len+=1
        #        len_+=1
        #    self.batch_acc.append(float(acc_)/len_*100.0)
        acc_=0
        len_=0
        results = []
        for gt_box,pred_box in zip(gt_boxes,reshaped_boxes):
            loc = ((pred_box[0]+pred_box[2])/2,(pred_box[1]+pred_box[3])/2)
            if loc[0]>gt_box[0] and loc[0]<gt_box[2] and loc[1]>gt_box[1] and loc[1]<gt_box[3]:
                acc_+=1
                results.append(1)
            else:
                results.append(0)
            len_+=1

        results = torch.from_numpy(np.array(results))
        out_dict['result'] = results 
        self.batch_acc.append(float(acc_)/len_*100.0)
        #    #for im_file in inp(inp["img_file"]):
        #    im = cv2.imread(im_file)
        #    im = cv2.rectangle(im, (int(gt_boxes[0]), int(gt_boxes[1])), (int(gt_boxes[2]), int(gt_boxes[3])), (255,0,0), 2)
        #    cv2.imwrite("test.jpg",im)
        ##print("Pred: ",reshaped_boxes)
        #print("GT: ",gt_boxes)
        out_dict['pred_boxes'] = reshaped_boxes
        out_dict['pred_scores'] = att_box_best
        # orig_annot = inp['orig_annot']
        # Sanity check
        # iou1 = (torch.diag(IoU_values(reshaped_boxes, orig_annot))
        #         >= self.acc_iou_threshold).float().mean()
        # assert actual_result.item() == iou1.item()
        return out_dict
        # return reduce_dict(out_dict)

    def get_eval_result(self, actual_bbox, annot, ids_to_use, msk=None):
        best_boxes = torch.gather(
            actual_bbox, 1, ids_to_use.view(-1, 1, 1).expand(-1, 1, 4))
        best_boxes = best_boxes.view(best_boxes.size(0), -1)
        if msk is not None:
            best_boxes[msk] = 0
        # self.best_boxes = best_boxes
        ious = torch.diag(IoU_values(best_boxes, annot))
        #print(self.acc_iou_threshold)
        # self.fin_results = ious
        return (ious >= self.acc_iou_threshold).float().mean(), best_boxes,ious

    def get_accuracy(self):
        mean_ = np.mean(self.batch_acc)
        std_ = np.std(self.batch_acc)
        self.batch_acc = []
        self.acc = 0.0
        self.len = 0.0
        return mean_,std_
        #return np.mean(self.batch_acc),np.std(self.batch_acc)
        #return (self.acc/self.len)*100.0


def get_default_eval(ratios, scales, cfg):
    return Evaluator(ratios, scales, cfg)
