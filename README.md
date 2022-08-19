# yolov7-instance-segmentation

## Features
- How to run Code on Windows
- How to run Code on Linux
- Frequently Asked Questions (FAQs)

## Coming Soon
- Development of streamlit dashboard for Instance-Segmentation with Object Tracking

## Requirements
- GPU (Needed for installation of detectron2)
- Git for Windows <a href="https://git-scm.com/download/win">Download Link</a>  
- Git on Linux (Install git on linux by using command in terminal. ```sudo apt-get install git```)

## Steps to run Code

### Linux Users
- Open the terminal and run mentioned command below to download & install anaconda for linux operating system
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh #(you will need to accept license terms from terminal, then you installation will continue)
```
- Once Anaconda Installed, restart your machine.

- Open the terminal in <B>home folder</B>, and run the mentioned command below.
```
cd ~
sudo chmod 777 .conda
```
- Clone the repository.
```
git clone https://github.com/RizwanMunawar/yolov7-instance-segmentation
```
- Goto the cloned folder.
```
cd yolov7-instance-segmentation
```
- Create envirnoment
```
conda env create -f environment.yml
```
- Activate the envirnoment
```
conda activate detectron2
```
- Install extra modules
```
pip install -r requirements.txt
```
- Download weights
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt
```
- Run Code with mentioned command.
```
#Basic Usage
python instance-segmentation.py

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python instance-segmentation.py --source "your IP Camera Stream URL"

#For WebCam
python instance-segmentation.py --source 0

#For External Camera
python instance-segmentation.py --source 1
```
- Output file will be created in the working directory with name ("your-file-name-without-extension"+"_segmentation.mp4")

### Window Users
- Download the <a href="https://repo.anaconda.com/archive/Anaconda3-2022.05-Windows-x86_64.exe">64-Bit</a> or <a href="https://repo.anaconda.com/archive/Anaconda3-2022.05-Windows-x86.exe">32-Bit</a> Anaconda <B>(Based on your system specifications)</B>.
- Install the executable
- Goto Start Menu and search for "Anaconda Prompt". Double Click to Open it.
- Change the path of anaconda prompt with mentioned command below.
```
cd "C:\Users\"yourusername"\Desktop
```
- Clone the repository.
```
git clone https://github.com/RizwanMunawar/yolov7-instance-segmentation
```
- Goto the cloned folder.
```
cd yolov7-instance-segmentation
```
- Create envirnoment
```
conda env create -f environment.yml
```
- Activate the envirnoment
```
conda activate detectron2
```
- Install extra modules
```
pip install -r requirements.txt
```
- Download weights from <a href="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt">link</a> and move them to the cloned folder.
- Run Code with mentioned command.
```
#Basic Usage
python instance-segmentation.py

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python instance-segmentation.py --source "your IP Camera Stream URL"

#For WebCam
python instance-segmentation.py --source 0

#For External Camera
python instance-segmentation.py --source 1
```
- Output file will be created in the working directory with name ("your-file-name-without-extension"+"_segmentation.mp4")

### RESULTS
<table>
  <tr>
    <td>Football Match Image Segmentation</td>
     <td>Cricket Match Image Segmentation</td>
    <td>FPS and Time Comparision Graph</td>
     </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185704342-59cb9bce-6be1-432b-90fc-2064feed4a67.png" width=640 height=240></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185706834-19ee1c9f-de91-439d-bba3-6b05c00be226.png" width=640 height=240></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185712079-e8ffcdfb-8d3b-467a-9620-d6186976370c.png" width=640 height=240></td>
  </tr>
  </tr>
 </table>
 
