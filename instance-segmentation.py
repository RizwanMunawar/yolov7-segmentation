import cv2
import yaml
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from models.experimental import attempt_load
from detectron2.structures import Boxes
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
from utils.general import non_max_suppression_mask_conf,strip_optimizer

@torch.no_grad()
def run(
        segweights='yolov7-mask.pt',
        source='football1.mp4',
        device='cpu'):
    
    #list to store time
    time_list = []
    
    #list to store fps
    fps_list = []
    
    #load hyp file for mask
    with open('hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    #load segmentation model
    model = attempt_load(segweights, map_location=device)  # load FP32 model
    _ = model.eval()

    #source path
    video_path = source

    #pass video to videocapture object
    cap = cv2.VideoCapture(video_path)

    #check if videocapture not opened
    if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

    #get video frame width
    frame_width = int(cap.get(3))

    #get video frame height
    frame_height = int(cap.get(4))

    #code to write a video
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{out_video_name}_segmentation.mp4",
                        cv2.VideoWriter_fourcc(*'mjpg'), 30,
                        (resize_width, resize_height))

    #count no of frames
    frame_count = 0
    
    #count total fps
    total_fps = 0 


    #loop until cap opened or video not complete
    while(cap.isOpened):

        print("Frame {} Processing".format(frame_count))

        #get frame and success from video capture
        ret, frame = cap.read()

        #if success is true, means frame exist
        if ret:
            
            #store frame
            orig_image = frame
            #convert frame to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, 640, stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.float()

            #start time for fps calculation
            start_time = time.time()
            
            #get output from image
            output = model(image)

            #get output coordinates for generation of masks
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

            #apply ROI poller
            pooler = ROIPooler(output_size=hyp['mask_resolution'], 
                            scales=(pooler_scale,), 
                            sampling_ratio=1, 
                            pooler_type='ROIAlignV2', 
                            canonical_level=2)

            #Apply NMS 
            output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, 
                                                                bases, 
                                                                pooler, 
                                                                hyp, 
                                                                conf_thres=0.25, 
                                                                iou_thres=0.6)

            #Draw segmented masks
            pred, pred_masks = output[0], output_mask[0]
            base = bases[0]
            print(base)
            bboxes = Boxes(pred[:, :4])
            original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
            pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
            pred_masks_np = pred_masks.detach().cpu().numpy()
            pred_cls = pred[:, 5].detach().cpu().numpy()
            pred_conf = pred[:, 4].detach().cpu().numpy()
            nimg = image[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)

            #convert image to original format
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
            pnimg = nimg.copy()

            #different color for each instance
            for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
                
                if conf < 0.25:
                    continue
                color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
                                                        
                pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
                pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            #Calculatio for FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            #append FPS in list
            fps_list.append(total_fps)
            
            #append time in list
            time_list.append(end_time - start_time)
            
            #add FPS on top of video
            cv2.putText(pnimg, f'FPS: {int(fps)}', (11, 100), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        else:
            break
    
    cap.release()
    # cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    
    #plot the comparision graph
    plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segweights', nargs='+', type=str, default='yolov7-mask.pt', help='yolov7 segmentation model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.segweights)
    main(opt)