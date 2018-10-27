# Utils
# Setting
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


# 미리 선언 -- unique, bbox_iou: write_results 내부에 사용됨
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res



def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou



def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    :param prediction: output (detection feature map)
    :param inp_dim: input 이미지 dimension
    :return: prediction, 2-D tensor -- 각 행은 bbox의 attribute를 나타냄
    """

    # 예
    # input_dim=416
    # 1st stride=32
    # 1st grid_size=13
    # num_classes=80

    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    # (batch_size, 13*13*3, 85)

    # detection feature map의 stride로 anchor를 나누어 줌
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    # anchors 변화
    # 원래: [(10, 13), (16, 30), (33, 23)] -- 10/32=0.3125
    # 이후: [(0.3125, 0.40625), (0.5, 0.9375), (1.03125, 0.71875)]

    # Sigmoid the  centre_X, centre_Y. and object confidence
    # 2, 3번째는 height와 width이므로 0~1사이일 필요 없음
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)    # size = (169, 1) when grid_size=13
    y_offset = torch.FloatTensor(b).view(-1,1)    # size = (169, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    # centre_X와 centre_Y 업데이트
    prediction[:, :, :2] += x_y_offset


    # Apply the anchors to the dimensions of the bounding box
    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors


    # Apply sigmoid activation to the the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 다시 원래 input_dim 대로 원상복귀 (stride=32만큼 곱함: 13X32 = 416)
    prediction[:, :, :4] *= stride

    return prediction





def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    :param prediction: (m, 10747, 85)
    :param confidence: objectness score threshold
    :param num_classes: Ex) 80
    :param nms_conf: NMS IoU threshold
    :return:
    """

    #1 Object Confidence Thresholding
    # prediction[:,:,4]는 object score를 담고 있는데 이게 confidence보다 크면 True를 반환하게 한다. (conf_mask)
    # True인 녀석만 남기고 나머지는 0이 되게 한다: prediction*conf_mask
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask


    #2 NMS
    # IoU를 편하게 계산하기 위해 centre_X, centre_Y, height, width를 코너 좌표로 바꿔준다.
    # new를 통해 같은 shape의 box_corner 텐서 생성
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]    # object confidence

    # 그런데 각 이미지마다 True detection의 수는 다를 수 있기 때문에
    # 과정은 각 이미지마다 한 번에 이루어져야 한다.

    batch_size = prediction.size(0)
    write = False

    # Loop 시작
    for index in range(batch_size):
        image_pred = prediction[index]  # 배치 안에 들어 있는 하나의 image Tensor
        # confidence thresholding
        # NMS

        # 80개의 class score 중 가장 높은 녀석을 찾아준다.
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (image_pred[:, 0:5], max_conf, max_conf_score)    # 길이 7
        # 업데이트한 대로 새로 합쳐줌(순서대로)
        image_pred = torch.cat(seq, dim=1)

        # 위에서 object confidence가 threshold보다 작은 bbox row는 0으로 만들어주었다.
        # 얘네를 없애자
        # image_pred[:4]는 (10647, 1)의 shape을 가졌고,
        # non_zero_ind는 여기서 zero가 아닌 녀석들의 indices를 저장한다.
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        # squeeze()는 shape 중 1인 부분을 없애준다.
        # 즉, 아래의 image_pred는
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # 만약 detect를 하나도 못했을 경우를 대비하여
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue


        # 이제 이미지에서 detected된 class를 가져오자
        img_classes = unique(image_pred_[:,-1]) # -1 index holds the class index

        # perform NMS -- cls: detections of 특정 class
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(index)
            # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class


            # output: [index of the image in the batch(1), 좌표(4), object score(1), max_conf_score(1), index of class(1)]
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0



# 클래스 이름을 담은 파일 로드
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names



def letterbox_image(img, inp_dim):
    """
    resize image with unchanged aspect ratio using padding
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas



def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

