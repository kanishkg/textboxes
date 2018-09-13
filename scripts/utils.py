import torch

def get_iou(box1, box2):
    left_top = torch.max(box1[:,:2], box2[:,:2])
    right_bot = torch.min(box1[:,2:], box2[:,2:])
    area_1 = torch.prod(a[:, 2:] - a[:, :2], dim=1)
    area_2 = torch.prod(b[:, 2:] - b[:, :2], dim=1)
    mask = (lt < rb).all(axis=2)
    intersection = torch.prod(rb - lt, axis=2) * mask
    union = area_1 + area_2 - intersection
    return intersection/union



class box_utility:
    def __init__(self):
        self.aspect_ratios = [1,2,3,5,7,10]
        self.input_dim = 300
        self.grids_size = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.sizes = [30, 60, 111, 162, 213, 264, 315]


    def create_prior_boxes(self):
        self.prior_boxes = []
        for s in range(len(self.grid_sizes)):
            for y in range(s):
                for x in range(s):
                    cx = (x+0.5)*self.steps[s]/300
                    cy = (y+0.5)*self.steps[s]/300
                    side = self.sizes[s]/300
                    self.prior_boxes.append([cx,cy,side,side])
                    side = np.sqrt(self.sizes[s]*self.sizes[s+1])/300
                    self.prior_boxes.append([cx,cy,side,side])
                    for a in self.aspect_ratios:
                        self.prior_boxes.append(cx,cy,side/np.sqrt(a),side*np.sqrt(a))
                        self.prior_boxes.append(cx,cy,side*np.sqrt(a),side/np.sqrt(a))
        self.prior_boxes = boxes = np.clip(np.array(self.prior_boxes),0.0,1.0)

    def get_prior_gt_match(self, boxes):
        self.modified_gt = {}
        for b,box in enumerate(boxes):
            for prior in self.prior_boxes:
                iou = get_iou(box, prior)
                max_iou = np.max(iou)
                if max_iou>thresh:
                    m = np.argmax(iou)
                    modified_gt[b] = prior


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
       The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
