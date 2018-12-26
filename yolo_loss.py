# Building YOLOv2 Loss Function

# Setting
#import os
#import sys
#sys.path.append(os.path.join(os.getcwd(), 'BasicNet'))

from yolo_config import *


class YOLO_LOSS(nn.Module):
    def __init__(self, config):
        super(YOLO_LOSS, self).__init__()
        self._config = config


    def forward(self, out, GT_bboxes):
        """
        :param out: Darknet의 최종 아웃풋,
                    shape = [None, NUM_BOX * (1 + 4 + NUM_CLASS), GRID_W, GRID_H ] = [None, 35, 13, 13]
        :param GT_bboxes: 실제 class와 bbox좌표를 담은 텐서,
                    shape = [None, 5], 예: [class, x, y, w, h] = [1.0000, 0.4848, 0.5118, 0.6617, 0.9764]
        :return: loss
        """
        _config = self._config
        lc = _config.lambda_coord
        ln = _config.lambda_noobj

        predicted = out.view(-1, _config.NUM_BBOX, _config.BBOX_SIZE, _config.GRID_W * _config.GRID_H)

        C_hat  = predicted[:, :, 0, :] # confidence
        xy_hat = predicted[:, :, 1:3, :]
        wh_hat = predicted[:, :, 3:5, :]
        p_hat  = predicted[:, :, 5:, :]

        xy     = GT_bboxes[:, 1:3]
        wh     = GT_bboxes[:, 3:5]

        #TODO 1) Mask: GT box가 없는 Cell의 경우 1<i,j>obj는 0이 된다.
        #TODO 2) 살아남은 cell들의 5개 bbox를 각각 GT_bbox와의 IoU를 계산하여 1등이 아닌 1<i,j>obj는 0이 된다.

        # iou



        wh_hat = torch.sqrt(wh_hat)
        wh     = torch.sqrt(wh)


        xy_loss = lc *

        """
        Args:
        preds: (tensor) model outputs, sized [batch_size,150,fmsize,fmsize].
        loc_targets: (tensor) loc targets, sized [batch_size,5,4,fmsize,fmsize].
        cls_targets: (tensor) conf targets, sized [batch_size,5,20,fmsize,fmsize].
        box_targets: (list) box targets, each sized [#obj,4].
        """

        preds = preds.view(batch_size, 5, 4+1+20, fmsize, fmsize)

        ### loc_loss
        xy = preds[:,:,:2,:,:].sigmoid()   # x->sigmoid(x), y->sigmoid(y)
        wh = preds[:,:,2:4,:,:].exp()
        loc_preds = torch.cat([xy,wh], 2)  # [N,5,4,13,13]

        # clas_targets의 dim=2인, 즉 class_probs에서 가장 높은 녀석을 찾으면
        # class일 확률이 가장 높은 녀석이 튀어나오고
        # index: [0]을 하면 index값은 버리고 실제 값만 나오게 된다. (tuple로 반환되기 때문에)
        # 그리고 squeeze를 해주면 [None, 5, 20, fmsize, fmsize]가 [None, 5, 1, fmsize, fmsize]가 되고
        # 또 이것이 [None, 5, 13, 13]이 된다.
        # 근데 여기서 0보다 큰 녀석을 찾아주면
        # 실제로 타겟이 있는 GT cell raw mask가 만들어진다.
        #
        pos = cls_targets.max(2)[0].squeeze() > 0  # [N,5,13,13]
        num_pos = pos.data.long().sum()
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,5,13,13] -> [N,5,1,13,13] -> [N,5,4,13,13]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)





# [None, 35, 13, 13]
# 35 = 5 X (5+2) = NUM_BOX * (1 + 4 + NUM_CLASS)










