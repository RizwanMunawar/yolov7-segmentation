# yolov7-instance-segmentation

## Coming Soon
- Development of streamlit dashboard for Instance-Segmentation with Object Tracking

## Steps to run Code

- Clone the repository
```
git clone https://github.com/RizwanMunawar/yolov7-segmentation.git
```
- Goto the cloned folder.
```
cd yolov7-segmentation
```
- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv yolov7seg
source yolov7seg/bin/activate

### For Window Users
python3 -m venv yolov7seg
cd yolov7seg
cd Scripts
activate
cd ..
cd ..
```
- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt) and store in "yolov7-segmentation" directory.
- 
- Run the code with mentioned command below.
```
python3 segment/predict.py --weights yolov7-seg.pt --source "video path.mp4"
```

- Output file will be created in the working directory with name ("yolov7-segmentation/runs/predict-seg/exp/"original-video-name.mp4")

### RESULTS
<table>
  <tr>
    <td>Football Match Image Segmentation</td>
     <td>Cricket Match Image Segmentation</td>
    <td>FPS and Time Comparision Graph</td>
     </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185704342-59cb9bce-6be1-432b-90fc-2064feed4a67.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185706834-19ee1c9f-de91-439d-bba3-6b05c00be226.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185712079-e8ffcdfb-8d3b-467a-9620-d6186976370c.png" width=640 height=180></td>
  </tr>
  </tr>
 </table>
 

## Custom Training
- Make sure to follow above mentioned steps before you will start training on custom dataset.
- Move your (segmentation custom labelled data) inside "yolov7-segmentation\data" folder with mentioned structure.

└── data

    └── train

        └── images (folder including all training images)
    
        └── labels (folder including all training labels)
  
    └── test
   
        └── images (folder including all testing images)
    
        └── labels (folder including all testing labels)

- Go to the <b>data</b> folder and create a file with name <b>custom.yaml</b> and paste the mentioned code below inside that.

```
train: "path to train folder"
val: "path to validation folder"

# number of classes
nc: 2

# class names
names: [ 'person','Bike']
```

- Go to the terminal, and run mentioned command below to start training.
```
python3 segment/train.py --data "custom.yaml" \
--batch 4 --weights '' --cfg yolov7-seg.yaml \
--epochs 10 --name yolov7-seg \
--img 640 --hyp hyp.scratch-high.yaml
```

## Testing
```
python test.py --data data/custom.yaml --img 256 --conf 0.25 --iou 0.65 --weights yolov7-seg.pt
```

## References
- https://github.com/WongKinYiu/yolov7/tree/u7/seg
- https://github.com/ultralytics/yolov5

## My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
