# yolov7-instance-segmentation

## Coming Soon
- Development of streamlit dashboard for Instance-Segmentation with Object Tracking

## Code Medium Blog
- https://chr043416.medium.com/train-yolov7-segmentation-on-custom-data-b91237bd2a29

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
- Download weights from [link](https://github.com/RizwanMunawar/yolov7-segmentation/releases/download/yolov7-segmentation/yolov7-seg.pt) and store in "yolov7-segmentation" directory.

- Run the code with mentioned command below.
```
#for segmentation with detection
python3 segment/predict.py --weights yolov7-seg.pt --source "videopath.mp4"

#for segmentation with detection + Tracking
python3 segment/predict.py --weights yolov7-seg.pt --source "videopath.mp4" --trk
```

- Output file will be created in the working directory with name <b>yolov7-segmentation/runs/predict-seg/exp/"original-video-name.mp4"</b>

### RESULTS
<table>
  <tr>
    <td>Car Semantic Segmentation</td>
     <td>Car Semantic Segmentation</td>
     <td>Person Segmentation + Tracking</td>
     </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/190402435-931f0ee3-9af1-4399-8222-1028d5afbd1a.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/190402752-521b7815-bea8-4cef-8b36-54fb7a962244.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/191729411-a8d8b5e2-bdbf-4c0e-bd1b-a52e23f7c9d3.png" width=640 height=180></td>
  </tr>
  </tr>
 </table>


## Custom Data Labelling

- I have used [roboflow](https://roboflow.com/) for data labelling. <b>The data labelling for Segmentation will be a Polygon box,While data labelling for object detection will be a bounding box</b>

- Go to the [link](https://app.roboflow.com/my-personal-workspace/createSample) and create a new workspace. Make sure to login with roboflow account.


![1](https://user-images.githubusercontent.com/62513924/190390384-db8f71fa-e963-4ee6-aaca-c49e993c64ae.png)


- Once you will click on <b>create workspace</b>, You will see the popup as shown below to upload the dataset.

![2](https://user-images.githubusercontent.com/62513924/190390882-fe08559d-ef47-450e-8613-2de899fffa4c.png)


- Click on upload dataset and roboflow will ask for workspace name as shown below. Fill that form and then click on <b>Create Private Project</b>
- Note: Make sure to select <b>Instance Segmentation</b> Option in below image.
 ![dataset](https://user-images.githubusercontent.com/62513924/190853038-612791d0-9b33-4222-b28a-63ac4c13ed83.png)


-You can upload your dataset now.

![Screenshot 2022-09-17 155330](https://user-images.githubusercontent.com/62513924/190853135-887b389c-2356-4435-a946-867bb05ac4f2.png)

- Once files will upload, you can click on <b>Finish Uploading</b>.

- Roboflow will ask you to assign Images to someone, click on <b>Assign Images</b>.

- After that, you will see the tab shown below.

![6](https://user-images.githubusercontent.com/62513924/190392948-90010cd0-ef88-437a-b94f-44ee93d8bc31.png)


- Click on any Image in <b>Unannotated</b> tab, and then you can start labelling.

- <b>Note:</b> Press p and then draw polygon points for <B>segmentation</b>

![10](https://user-images.githubusercontent.com/62513924/190394353-d7dd7b7f-7a07-4738-99b6-1d5ae66b5bca.png)


- Once you will complete labelling, you can then export the data and follow mentioned steps below to start training.

## Custom Training

- Move your (segmentation custom labelled data) inside "yolov7-segmentation\data" folder by following mentioned structure.



![ss](https://user-images.githubusercontent.com/62513924/190388927-62a3ee84-bad8-4f59-806f-1185acdc8acb.png)



- Go to the <b>data</b> folder, create a file with name <b>custom.yaml</b> and paste the mentioned code below inside that.

```
train: "path to train folder"
val: "path to validation folder"
# number of classes
nc: 1
# class names
names: [ 'car']
```

- Go to the terminal, and run mentioned command below to start training.
```
python3 segment/train.py --data data/custom.yaml --batch 4 --weights '' --cfg yolov7-seg.yaml --epochs 10 --name yolov7-seg --img 640 --hyp hyp.scratch-high.yaml
```

## Custom Model Detection Command
```
python3 segment/predict.py --weights "runs/yolov7-seg/exp/weights/best.pt" --source "videopath.mp4"
```

## RESULTS
<table>
  <tr>
    <td>Car Semantic Segmentation</td>
     <td>Car Semantic Segmentation</td>
     <td>Person Segmentation + Tracking</td>
     </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/190402435-931f0ee3-9af1-4399-8222-1028d5afbd1a.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/190410343-ada838c6-e505-4248-8a76-fbc5996e091e.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/191729411-a8d8b5e2-bdbf-4c0e-bd1b-a52e23f7c9d3.png" width=640 height=180></td>
  </tr>
  </tr>
 </table>


## References
- https://github.com/WongKinYiu/yolov7/tree/u7/seg
- https://github.com/ultralytics/yolov5

## My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/how-can-computer-vision-products-help-in-warehouses-aa1dd95ec79c
- https://medium.com/augmented-startups/problems-in-the-development-of-computer-vision-products-eb081ec7aa2e
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://chr043416.medium.com/train-yolov7-segmentation-on-custom-data-b91237bd2a29
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
