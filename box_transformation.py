# Utils
# Setting
from yolo_config import *


# Config
C = 2
THRESHOLD = 0.6



def box_to_corner(box1, box2):
    """
    bbox의 [x, y, w, h]를 [x1, y1, x2, y2]로 변환해 줍니다.
    """
    b1_x, b1_y, b1_w, b1_h = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x, b2_y, b2_w, b2_h = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    b1_x1, b1_x2 = b1_x - b1_w/2, b1_x + b1_w/2
    b1_y1, b1_y2 = b1_y - b1_h/2, b1_y + b1_h/2

    b2_x1, b2_x2 = b2_x - b2_w/2, b2_x + b2_w/2
    b2_y1, b2_y2 = b2_y - b2_h/2, b2_y + b2_h/2

    return b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2



def calculate_bbox_iou(box1, box2):
    """
    2개의 bbox의 IoU를 계산한다.
    """
    # corner화 된 좌표를 가져온다.
    b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2 = box_to_corner(box1, box2)

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) *\
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou



def calculate_multiple_bbox_iou(box1, box2):
    """
    pair-bbox 세트의 IoU를 한 번에 계산한다.
    """
    # corner화 된 좌표를 가져온다.
    b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2 = box_to_corner(box1, box2)

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) *\
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou



def filter_by_confidence(confidence, boxes, class_probs, threshold=0.6):
    """
    confidence가 threshold보다 낮은 경우 제거해준다.
    남은 confidence와 class_probs를 곱하여 class_scores를 생성한다.

    :param confidence: (None, 1)
    :param boxes: (None, 4)
    :param class_probs: (None, C)
    :param threshold: 필터링 threshold
    """

    confidence[confidence < threshold] = 0.0

    # 아래 2줄로 confidence가 0이 된 bbox 세트는 제거해준다.
    i, _ = torch.unbind(confidence.nonzero(), 1)
    confidence, boxes, class_probs = confidence[i], boxes[i], class_probs[i]

    # 브로드캐스팅을 통해 confidence와 class_probs를 곱하여 class_scores 생성
    # class_scores.shape = (None, C)
    class_scores = confidence * class_probs

    return boxes, class_scores


# 여기서 class_scores는 이미 P(Object) * IoU 필터링 이후
# confidence와 class_prob_scores를 곱한 수치이다.

# boxes = last_tensor[..., 1:5]
# class_scores = last_tensor[..., 6:8]



def nms(boxes, class_scores, threshold=0.6):
    """
    :param boxes: bbox 좌표, (None, 4)
    :param class_scores: confidence * class_prob_scores, 클래스 별 score, (None, C)
    :param threshold: NMS Threshold
    """

    for class_number in range(C):
        target = class_scores[..., class_number]

        # 현재 class 내에서 class_score를 기준으로 내림차순으로 정렬한다.
        sorted_class_score, sorted_class_index = torch.sort(target, descending=True)

        # idx: 아래 bbox_max의 Index
        # bbox_max_idx: 정렬된 class_scores를 기준으로 가장 큰 score를 가지는 bbox Index
        for idx, bbox_max_idx in enumerate(list(sorted_class_index.numpy())):
            # 기준 class_score가 0이라면 비교할 필요가 없다.
            # 아래 threshold 필터링에서 0으로 바뀐 값이기 때문이다.
            if class_scores[bbox_max_idx, class_number] != 0.0:
                # 0이 아니라면 순서대로 criterion_box로 지정된다.
                bbox_max = boxes[bbox_max_idx, :]

                # criterion_box가 아닌 다른 box들을 리스트로 미리 지정한다.
                #others = [index for index in list(sorted_class_index.numpy()) if index != i]
                others = list(sorted_class_index.numpy())[idx+1:]

                # 비교 대상 box들에 대해서
                # bbox_cur_idx: 비교 대상 bbox Index
                for bbox_cur_idx in others:
                    bbox_cur = boxes[bbox_cur_idx, :]
                    iou = calculate_bbox_iou(bbox_max, bbox_cur)
                    # print(bbox_max_idx, bbox_cur_idx, iou)

                    # iou가 threshold를 넘으면 (기준 box와 비교 대상 box가 너무 많이 겹치면)
                    # 그 해당 box의 현재 class의 class_score를 0으로 만들어 준다.
                    if iou > threshold:
                        class_scores[bbox_cur_idx, class_number] = 0.0

    return boxes, class_scores


def box_transformation_test():
    
    c = np.array([[1.0], [1.0], [1.0], [1.0], [0.3]])
    a = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1], [0.79, 0.79, 0.1, 0.1], [0.6, 0.6, 0.2, 0.4]])
    s = np.array([[0.7, 0.1], [0.5, 0.4], [0.4, 0.3], [0.6, 0.1], [0.1, 0.1]])
    
    confidence = torch.from_numpy(c)
    boxes = torch.from_numpy(a)
    class_probs = torch.from_numpy(s)
    
    print(confidence, boxes, class_probs)
    boxes, class_scores = filter_by_confidence(confidence, boxes, class_probs, 0.6)
    new_boxes, new_class_scores = nms(boxes, class_scores, 0.6)
    print('new_boxes:\n', new_boxes)
    print('new_class_scores:\n', new_class_scores )




