# Fast Four Cognitive Architecture
## Team Fast Four: Jasper Ding, Son Pham, Tung Phan, Keyi Zhang
## Bucknell University Introduction to AI & Cognitive Science CSCI379 2017
## Professor Christopher Dancy

In this project, we aim to build a neuro-evolution agent that can play 
Top Gear (SNES 1992) game.

# Warning

* This repo instruction asumes you are running everything on a Windows machine as the BizHawk build we provided is compiled for Windows platform. If you want to run it on Linux/Mac OS, please refer to the compile guide for BizHawk at [https://github.com/TASVideos/BizHawk](https://github.com/TASVideos/BizHawk)
* Make sure that port 37979 on your localhost is available and you have proper access permission.

# Installation
We have built a customized BizHawk binary in the ```output``` folder. You don't have to build it by yourself. We have also included all the checkout points for ```neat``` training so that you can try out each individual generation stage.

Most of the programs run on Python. As a result, you need to install all the Python packages. To do so, you have to use ```pip```. It is highly recommended to use ```virtualenv``` to separate your working environment from your system's default environment. To make things easier, we have also included pip ```requirements.txt```. The following commands assume that you are using a Windows machine with ```Python 3``` added to its environment path.
```
$ virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Setting up webcam
By default it assumes that your machine has two webcams. As a result, it will try to access the second webcam (index 1). If it is not the case, you need to change the webcam index in the source code. Open ```capture.py``` and find line ```110```. Then change ```cv2.VideoCapture(1)``` to any camera number you plan to use.

# Running


## Training NEAT Agent

There are 2 different components to set this training process.
We list those in order of running:

* Run ```./TrainingServer.py```
* Run executable "./output/EmuHawk.exe" to start the modified BizHawk emulator. 
Load up the desirable ROM. For our project, it is ```./Top Gear (USA).sfc```.

## Heuristics Expert System, with Webcam

There are 3 different components to set this training process.
We list those in order of running:

* Run ```./HeuristicServer.py```.
* Run ```./capture.py```, adjust your webcam and mark the corner points of the game
screen
* Run executable ```./output/EmuHawk.exe``` to start the modified BizHawk emulator. 
Load up the desirable ROM. For our project, it is ```./Top Gear (USA).sfc```.
Load a desirable save state. In this project, F2 key is for Game Bot, F3 key is
for Human player using Game Controller.

On a side note, you can play this game with your head if you use a webcam that faces directly towards you.

## Running NEAT Agent, with Webcam

There are 4 different components to run this agent set up. 
We list those in order of running:

* Edit ```./InterfaceServer.py```, make sure that constant BIZHAWK is False. 
Run ```./InterfaceServer.py```.

* Edit ```./NeuralNetwork.py```, make sure that CHOSEN_GEN and CHOSEN_GENOME 
refer to the correct NEAT generation/genome. Our best performing combination 
is CHOSEN_GEN = 10, CHOSEN_GENOME = 1. Run ```./NeuralNetwork.py```

* Run executable ```./output/EmuHawk.exe``` to start the modified BizHawk emulator. 
Load up the desirable ROM. For our project, it is ```./Top Gear (USA).sfc```.
Load a desirable save state. In this project, F2 key is for Game Bot, F3 key is
for Human player using Game Controller.

* Run ```./capture.py```, adjust your webcam and mark the corner points of the game
screen


## Running NEAT Agent, without Webcam

There are 3 different components to run this agent set up. 
We list those in order of running:

* Edit ```./InterfaceServer.py```, make sure that constant BIZHAWK is True. 
Run ```./InterfaceServer.py```

* Edit ```./NeuralNetwork.py```, make sure that CHOSEN_GEN and CHOSEN_GENOME 
refer to the correct NEAT generation/genome. Our best performing combination 
is CHOSEN_GEN = 10, CHOSEN_GENOME = 1. Run ```./NeuralNetwork.py```

* Run executable ```./output/EmuHawk.exe``` to start the modified BizHawk emulator. 
Load up the desirable ROM. For our project, it is ```./Top Gear (USA).sfc```.
Load a desirable save state. In this project, F2 key is for Game Bot, F3 key is
for Human player using Game Controller.

