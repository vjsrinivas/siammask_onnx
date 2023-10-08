# this script is meant to be placed in SiamMask/ root folder

import sys
import os
from typing import Tuple
from types import SimpleNamespace
import torch
import torch.nn as nn
from utils.anchors import Anchors
from utils.config_helper import load_config
from utils.load_helper import load_pretrain
from loguru import logger
from tools.test import generate_anchor
import glob
import numpy as np
import cv2
import torch.nn.functional as F

# custom path appended to PYTHON PATH for resnet.py
sys.path.append("experiments/siammask_sharp")
from experiments.siammask_sharp.custom import ResDown, UP, MaskCorr


class Custom(nn.Module):
    def __init__(self, anchors):
        super(Custom, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)

        self.features = ResDown()
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

    def track_mask(self, search: torch.Tensor, zf: torch.Tensor):
        feature, _search = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(zf, _search)
        corr_feature = self.mask_model.mask.forward_corr(zf, _search)
        pred_mask = self.mask_model.mask.head(corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask, feature, corr_feature

    def track_refine(
        self,
        pos: torch.Tensor,
        feature: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        corr_feature: torch.Tensor,
    ):
        pred_mask = self.refine_model(feature, corr_feature, pos)
        return pred_mask

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.v1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.v2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.h0 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

        for modules in [
            self.v0,
            self.v1,
            self.v2,
            self.h2,
            self.h1,
            self.h0,
            self.deconv,
            self.post0,
            self.post1,
            self.post2,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(
        self,
        f: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        corr_feature: torch.Tensor,
        pos: torch.Tensor,
    ):
        p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[
            :, :, 4 * pos[0] : 4 * pos[0] + 61, 4 * pos[1] : 4 * pos[1] + 61
        ]
        p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[
            :, :, 2 * pos[0] : 2 * pos[0] + 31, 2 * pos[1] : 2 * pos[1] + 31
        ]
        p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[
            :, :, pos[0] : pos[0] + 15, pos[1] : pos[1] + 15
        ]
        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params

def unravel_indices(
    indices: torch.LongTensor,
    shape: torch.Tensor,
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for i in range(len(shape)-1,-1,-1):
        dim = shape[i]
        coord.append(indices % dim)
        indices = indices // dim
    coord = torch.stack(coord[::-1], dim=-1)
    return coord

def unravel_index(
    indices: torch.LongTensor,
    shape: torch.Tensor,
) -> torch.Tensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """

    return unravel_indices(indices, shape).int()
    #return tuple(coord)

class WrapperCustom(nn.Module):
    def __init__(self, model:Custom, hyp:dict, device:str, target_sz) -> None:
        super().__init__()
        self.model = model
        self.model.to(device)

        #### MANUAL P CREATION ####:
        self.exemplar_size = 127  # input z size
        self.instance_size = 255  # input x size (search region)
        self.total_stride = 8
        self.context_amount = 0.5  # context amount for the exemplar
        self.window_influence = 0.4
        self.lr = 1.0
        self.out_size = 63  # for mask
        base_size = 8
        self.penalty_k = 0.04
        self.score_size = (self.instance_size-self.exemplar_size)//self.total_stride+1+base_size
        self.unravel_mask = torch.Tensor([5, self.score_size, self.score_size])
        self.scales = self.model.anchors['scales']
        self.ratios = self.model.anchors['ratios']
        self.anchor_num = self.model.anchor_num
        self.anchor = torch.Tensor(generate_anchor(self.model.anchors, self.score_size)).to(device)
        self.update(hyp, self.model.anchors)
        self.renew()

        self.window = torch.Tensor(np.outer(np.hanning(self.score_size), np.hanning(self.score_size)))
        self.window = torch.Tensor(torch.tile(self.window.flatten(), (self.anchor_num,) )).to(device)
        
        self.device = device
        self.target_sz = torch.Tensor(target_sz)
        wc_z = target_sz[0] + self.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self.context_amount * sum(target_sz)
        self.s_z = torch.round(np.sqrt(wc_z * hc_z))

    def forward(self, x:torch.Tensor, target_pos:torch.Tensor, target_sz:torch.Tensor, scale_x:float, z_features:torch.Tensor, first_call:bool):
        if first_call:
            _x = x[:,:,:127,:127]
            z_out = self.model.features(_x)
            return torch.zeros((127,127), dtype=torch.float32).to(self.device), target_pos, target_sz, z_out, torch.zeros((2)).long()

        score, delta, mask, feature, corr_feature = self.model.track_mask(x, z_features)

        # Post-processing for the refinement module:
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1)[:,1]

        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0]
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1]
        delta[2, :] = torch.exp(delta[2, :]) * self.anchor[:, 2]
        delta[3, :] = torch.exp(delta[3, :]) * self.anchor[:, 3]

        target_sz_in_crop = self.target_sz*scale_x
        s_c = self.change(self.sz(delta[2, :], delta[3, :]) / (self.sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = self.change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty
        
        penalty = torch.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore = penalty * score

        pscore = pscore * (1 - self.window_influence) + self.window * self.window_influence
        best_pscore_id = torch.argmax(pscore)
        pred_in_crop = delta[:, best_pscore_id] / scale_x
        lr = penalty[best_pscore_id] * score[best_pscore_id] * self.lr  # lr for OTB
        
        res_x = pred_in_crop[0] + target_pos[0]
        res_y = pred_in_crop[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        target_pos[0] = res_x
        target_pos[1] = res_y
        target_sz[0] = res_w
        target_sz[1] = res_h
        
        best_pscore_id_mask = unravel_index(best_pscore_id, self.unravel_mask)
        delta_yx = best_pscore_id_mask[[2,1]].long() # y,x tensors

        # refinement
        mask = self.model.track_refine( delta_yx, feature, corr_feature).sigmoid().squeeze().view(self.out_size, self.out_size)
        return mask, target_pos, target_sz, z_features, delta_yx

    def change(self, r):
        return torch.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return torch.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return torch.sqrt(sz2)
    
    def get_subwindow_tracking(self, im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = torch.round(pos[0] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = torch.round(pos[1] - c)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        # zzp: a more easy speed version
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), dtype=np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        im_patch = torch.Tensor(im_patch)

        return self.im_to_torch(im_patch) if out_mode in 'torch' else im_patch

    def im_to_torch(self, img):
        img = torch.permute(img, (2, 0, 1))  # C*H*W
        return img
    
    ### MANUAL P SCRIPTING ###
    def update(self, newparam=None, anchors=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
        if anchors is not None:
            if isinstance(anchors, dict):
                anchors = Anchors(anchors)
            if isinstance(anchors, Anchors):
                self.total_stride = anchors.stride
                self.ratios = anchors.ratios
                self.scales = anchors.scales
                self.round_dight = anchors.round_dight
        self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + self.base_size
        self.anchor_num = len(self.ratios) * len(self.scales)


if __name__ == '__main__':
    logger.info("Exporting SiamMask to ONNX")
    logger.info("Creating model")
    # Parameter setup
    args = SimpleNamespace(
        config="experiments/siammask_sharp/config.json",
        resume="experiments/siammask_sharp/SiamMask_VOT.pth"
    )
    cfg = load_config(args)
    _model = Custom(anchors=cfg['anchors'])
    _model = load_pretrain(_model, args.resume)

    logger.debug("Loading in images...")
    img_files = sorted(glob.glob(os.path.join("./data/tennis", '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    logger.info("Meta data information and wrapper object")
    x,y,w,h = 301, 104, 179, 264
    im = ims[0]
    target_sz = torch.Tensor([w, h])
    model = WrapperCustom(_model, cfg['hp'], "cuda:0", target_sz)
    model.eval()
    avg_chans = np.mean(im, axis=(0, 1))
    target_pos = torch.Tensor([x + w / 2, y + h / 2])

    logger.info("Exporting")
    with torch.no_grad():
        
        logger.info("Input for initialization phase of model (branch 1)")
        z_crop = torch.zeros((1,3,model.instance_size,model.instance_size))
        _z_crop = model.get_subwindow_tracking(im, target_pos, model.exemplar_size, torch.round(model.s_z), avg_chans).unsqueeze(0)
        z_crop[:,:,:127,:127] = _z_crop
        z_crop = z_crop.to("cuda:0")
        dummy_z_feat = torch.zeros((1,256,7,7)).to("cuda:0")

        logger.info("Input for prediction phase of model (branch 2)")

        im = torch.Tensor(ims[1])
        wc_x = target_sz[1] + model.context_amount * sum(target_sz)
        hc_x = target_sz[0] + model.context_amount * sum(target_sz)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = model.exemplar_size / s_x
        d_search = (model.instance_size - model.exemplar_size) / 2
        pad = d_search / scale_x
        s_x = s_x + 2 * pad

        x_crop = model.get_subwindow_tracking(im, target_pos, model.instance_size, torch.round(s_x), avg_chans).unsqueeze(0)
        x_crop = x_crop.to("cuda:0") 

        logger.info("Torchscripting...")
        input1 = (x_crop, target_pos, target_sz, scale_x, dummy_z_feat, False)
        input0 = (z_crop, target_pos, target_sz, scale_x, dummy_z_feat, True)
        scriptedModel = torch.jit.script(model, example_inputs=[input0, input1])

        # Export the model
        torch.onnx.export(scriptedModel,               # model being run
                    input1,                         # model input (or a tuple for multiple inputs)
                    "siammask_vot.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=17,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['im', 'target_pos0', 'target_sz0', 'scale_x', 'z_features0', 'first_time'],   # the model's input names
                    output_names = ['output', 'target_pos1', 'target_sz1', 'z_features1', 'delta_yx'], # the model's output names
                    keep_initializers_as_inputs=True
        )

        print("============================================================")

        logger.info("Simplifying model")
        import onnx
        import onnxoptimizer
        onnxModel = onnx.load("siammask_vot.onnx")
        onnx.checker.check_model(onnxModel)

        all_available_passes = onnxoptimizer.get_available_passes()
        fuse_and_elimination_passes = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized_model = onnxoptimizer.optimize(model=onnxModel, passes=fuse_and_elimination_passes, fixed_point=False)
        onnx.save(proto=optimized_model, f="siammask_vot_simp.onnx")

        '''
        onnxModel_simp, check = simplify(onnxModel)
        print(check)
        onnx.save(onnxModel_simp, "siammask_vot_simp.onnx")
        '''