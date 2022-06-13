# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""
import numpy as np
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEdepth = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # cp = 1, cn = 0, 正负样本的损失权重
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox = torch.zeros(1, device=device), torch.zeros(1, device=device)
        lobj, ldepth = torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, tdepth = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # pi.shape = (bs, na, w, h, 4+1+class_num + 1)
                # ps.shape = (220, 4+1+class_num + 1)， 某次运行中的??是220
                # 对应targets的预测值，pi是每一层的预测tensor
                # gj,gi是中心点所在feature map位置，是采用targets中心点所在位置的anchor来回归。

                # Regression
                # 将预测值转换成box，并计算预测框和gt框的iou损失。
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 定位激活公式，anchors表示3个检测层anchor
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # tobj.shape = (bs, 3, w, h), if model.gr=1, then tobj is the iou with shape(bs, 3, w, h)
                # tobj是shape=(bs, 3, w, h)的tensor,
                # 正样本(anchor)处保存预测框和gt框的iou，负样本(anchor)处仍然是0，用作obj损失的真值
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t.shape = (220, cls_num)，t的值都是cn(0)
                    t = torch.full_like(ps[:, 5:5+self.nc], self.cn, device=device)  # targets
                    # 为正样本分配标签，cp=1，变成onehot类型标枪。 t.shape = (220, cls_num)
                    # tcls[i].shape = (220,)， tcls[i]里面所有的值都是0~cls_num
                    t[range(n), tcls[i]] = self.cp
                    # ps.shape = (220, 4+1+class_num + 1)， ??=220
                    # 只有正样本的分类损失
                    ps_t = ps[:, 5:5+self.nc]
                    lcls += self.BCEcls(ps[:, 5:5+self.nc], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # depth  # pre_depth = ps[:, -1]
                mse_l = torch.nn.MSELoss()
                # ldepth += mse_l(ps[:, -1], tdepth[i])
                ldepth += torch.log(1 + mse_l(ps[:, -1], tdepth[i]))

                # L1_l = torch.nn.L1Loss()
                # ldepth += L1_l(ps[:, -1], tdepth[i])  # L1损失在视差不连续处是鲁棒的并且对异常值或噪声具有低灵敏度
            # only balance P3-P5; iou of preds and targets is the target of BCE.
            # pi[..., 4] 是前景的预测值，‘1’代表肯定是前景。这里单独计算一个前景预测的损失，还要乘以一个系数。
            # lobj损失包括正负样本的损失，如此训练，使得负样本处的预测值接近0.
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        d_gain = self.hyp['depth'] if self.hyp.get('anchors') else 1
        ldepth *= d_gain
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + ldepth) * bs, torch.cat((lbox, lobj, lcls, ldepth)).detach()

    def build_targets(self, p, targets):
        # p的内容 [batch, num_anchor=3, h, w, nc+6], [] , [] 3 个 tensor
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)  ---depth
        # na是每一层layer的每一个位置的anchor的数量，nt是当前batch图片中所有目标框的数量
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tdepth = [], [], [], [], []
        gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
        # ai is matrix: [[0,0,...,0], [1,1,...,1], [2,2,...,2]], ai.shape = (na, nt) 生成一个ai，shape为（na,nt)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # after repeat, targets.shape = (na, nt, 6), after cat, targets.shape = (na, nt, 7)
        # 将targets扩充至(na, nt, 7)，也就是每个anchor与每个targets都有对应，为了接下来计算损失用
        # targets的值[[[image,class,x,y,w,h,0,depth],      ---depth
        #             [image,class,x,y,w,h,0,depth],   ---depth
        #               	...		共nt个   ]
        # 			  [[image,class,x,y,w,h,1,depth]，  ---depth
        #              [image,class,x,y,w,h,1,depth],  ---depth
        #                   ...		共nt个    ]
        # 			  [[image,class,x,y,w,h,2,depth]，  ---depth
        #              [image,class,x,y,w,h,2,depth],  ---depth
        #                   ...		共nt个    ]
        #          ]
        # ai[:, :, None]将ai扩展成shape（na, nt, 1）
        # targets.repeat(na, 1, 1) 将target在深度channel上扩展为na即 target.shape (a, b)->(na, a, b)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 每一层layer(共三层）单独计算。
        # 先通过判定每个target和3个anchor的长宽比是否满足一定条件，来得到满足条件的anchor所对应的targets (t)。
        # 这时的anchor数量是3，并不是某个位置的anchor，而是当前层的anchor。
        # 这时的t是3个anchor对应的targets的值，也就是说如果一个target如果对应多个anchor,那么t就有重复的值。

        # 然后根据t的每个target的中心点的偏移情况，得到扩充3倍的t。
        # 这时的t就是3个anchor对应的targets的值的扩充。

        # 接下来indices保存每层targets对应的图片索引，对应的anchor索引（只有3个），以及中心点坐标。
        # 接下来计算损失的时候，要根据targets对应的anchor索引来选择在某个具体位置的anchors,用来回归。

        for i in range(self.nl):  # nl: number of detection layers (P3-P5)
            # 每一层的anchor数量都是3个，三层共9个。
            anchors = self.anchors[i]
            # p[i].shape = (bs, na, h, w, 4+num_class+1)
            # gain[2:6]: [w, h, w, h], w & h is the width and height of the feature map
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors, before mul gain, targets is normalized. t.shape = (na, nt, 7)
            # 将提前归一化的targets转变成当前feature map下的绝对尺寸
            t = targets * gain
            if nt:  # number of target == target.shape[0]
                # Matches, t[:, :, 4:6].shape=(na, nt, 2), anchors[:, None].shape=(3, 1, 2)
                # 在当前feature map下，绝对尺寸的targets / 绝对尺寸的anchor，得到所有目标对于所有anchor的长宽的比例。
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # max of w & h ratio between targets and anchor that < 4.
                # it means that the targets match the anchors. j.shape = (na, nt)
                # 根据得到的比例来判定targets和当前layer的3个anchor是否匹配，最大边长的长度比小于4，则认为匹配上了。
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # before filter, t.shape = (3, nt, 7), j.shape = (3, nt)
                # after filter, t.shape = (?, 7)
                # t经过[j]索引后，就是匹配上这3个anchor的targets，匹配的数量视情况而定，这里用？表示。
                # t经过[j]索引后的值有重复，因为每个target可能匹配到多个anchor
                # 到这一步其实指示根据anchor的长宽比来计算匹配的targets。
                t = t[j]  # filter  筛选anchor与ground truth比值小于4的匹配目标

                # Offsets
                # gxy.shape = (?, 2), it means gt's x, y
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                # j means if x lean in left, k means if y lean in top.
                # l means if x lean in right, m means if y lean in bottom.
                # 得到该点更向哪个方向偏移，j.shape = (?,)，其他k,l,m的shape一致。  ###　正样本的定义！
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j.shape = (5, ?), ? means nrof filtered anchors
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 在[j]索引之前，t.shape = (5, ?, 7),
                # 5 means the 4+1 grids, which '1' is the center grid, '4' means the surounding grids.
                # 在[j]索引之后，t.shape = (??, 7)， 得到了扩充了的偏移后的targets。
                # 除了targets的中心点落在在feature map上的grid位置外，还有'上下左右'四个方向中，两个更靠近的grid。
                t = t.repeat((5, 1, 1))[j]
                # shape(1, ?, 2) + shape(5, 1, 2) = shape(5, ?, 2)
                # after indexed by [j], offsets.shape = (??, 2),
                # 举例来说，gxy.shape = (72, 2), offsets.shape = (220, 2)。
                # shape=(5, 72, 2)的tensor经过[j]的索引后，offsets.shape = (220, 2)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # which grid does the gt falls in. gij.shape = (??, 2), ?? ≈ ?*3
            # 对所有targets和match anchor的坐标集合做一个补充，不光是match的那个anchor的grid坐标，
            # 还包括所有targets在’上下左右‘四个方向中，更靠近的两个方向的位置的坐标。
            gij = (gxy - offsets).long()
            # 拆成横坐标和纵坐标，这些就代表正样本的坐标
            gi, gj = gij.T  # grid xy indices

            d = t[:, 6]  # 深度值

            # Append
            a = t[:, 7].long()  # anchor indices (0或者1或者2）, a.shape = (??,) 扩展的的输出的layer indices（大中小）

            # indices将每层的正样本的图片索引号(一个batch中的图片的索引号）
            # 对应的anchor索引号(只有3个)，以及坐标保存下来。
            # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            # tbox是[(??, 4), (??, 4), (??,4)]的list, 保存了每个正样本对应的gt box？
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # anch是[(??, 2), (??, 2), (??, 2)]的list，保存了每个正样本anchor。
            anch.append(anchors[a])  # anchors
            # tcls是[(??,), (??,), (??,)]的list, 保存了每个正样本对应的类别
            tcls.append(c)  # class
            # tdepth是[(??,), (??,), (??,)]的list, 保存了每个正样本对应的深度
            tdepth.append(d)

        return tcls, tbox, indices, anch, tdepth
