import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from detectron2.structures import Boxes
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
from utils.general import non_max_suppression_mask_conf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('hyp.scratch.mask.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
    
weigths = torch.load('yolov7-mask.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()

image = cv2.imread('./1.png')  # 504x378 image
image = letterbox(image, 640, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
image = image.to(device)
image = image.half()
output = model(image)
inf_out, train_out, attn, mask_iou, bases, sem_output = (output['test'],
                                                        output['bbox_and_cls'], 
                                                        output['attn'], 
                                                        output['mask_iou'], 
                                                        output['bases'], 
                                                        output['sem'])
bases = torch.cat([bases, sem_output], dim=1)
nb, _, height, width = image.shape
names = model.names
pooler_scale = model.pooler_scale
pooler = ROIPooler(output_size=hyp['mask_resolution'], 
                   scales=(pooler_scale,), 
                   sampling_ratio=1, 
                   pooler_type='ROIAlignV2', 
                   canonical_level=2)

output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, 
                                                    bases, 
                                                    pooler, 
                                                    hyp, 
                                                    conf_thres=0.25, 
                                                    iou_thres=0.6)

pred, pred_masks = output[0], output_mask[0]
base = bases[0]
bboxes = Boxes(pred[:, :4])
original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
pred_masks_np = pred_masks.detach().cpu().numpy()
pred_cls = pred[:, 5].detach().cpu().numpy()
pred_conf = pred[:, 4].detach().cpu().numpy()
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
pnimg = nimg.copy()

for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
    if conf < 0.25:
        continue
    color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
                        
                        
    pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
    pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)