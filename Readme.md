TopSideMonitor: fish behavior recording and 3D tracking
====================================================

TopSideMonitor is a python program used for recording and anylysis of zebrafish olfactory conditioning behavior (a paper in preparation). TopSideMonitor uses two webcams (or FireWire camera) to track fish in 3D (Top view and Side view). TopSideMonitor can optionally use Theano/lasagne based deep learning model to head and chest coordinate estimation (for orientation).

#Installation

Tested on anaconda Python 2.7 64 bit series on Windows 7. 

Install dependencies: numpy, scipy, scikit-image, wxpython, matplotlib, xlrd, xlwt, opencv

This can be done by miniconda command below:

```
(using Miniconda3 as example)

conda create -n topside python=2.7 matplotlib scipy scikit-image wxpython xlrd xlwt

activate topside (or source activate topside on Linux/Mac)

conda install -c https://conda.anaconda.org/menpo opencv3
```

(Optional, for a FireWire camera)

Install motmot.cam_iface from http://code.astraw.com/projects/motmot/.

(Optional, for lasagne based deep learning tracking model)

Instructions here are for without GPU support. For GPU, you will need to follow instructions given in Theano website for CUDA toolkit/GPU support before installing theano/lasagne (http://deeplearning.net/software/theano/install.html). GPU is almost necessary for training phase (x10-x100 times faster with CUDA compute capability 3.0 and above) but may not be needed for test/prediction phase. 

```
activate topside (or source activate topside on Linux/Mac)

conda install mingw libpython

pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt

pip install https://github.com/Lasagne/Lasagne/archive/master.zip

pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
```

At the time of writing, on Windows 7, these command will install nolearn 0.6a0.dev0 (not 0.7.git), Theano 0.8.0, Lasagne 0.2.dev1, scikit-learn 0.17.1.

And then, on Windows, install TDM GCC as explained in http://deeplearning.net/software/theano/install_windows.html


#How to use

* Recording (TopSideMonitor_FrontEnd.py)

  - This is the GUI frontend for recording fish behavior. Run this script to get the GUI and set parameters and press [Launch] button to open two camera frame viewers.
   
  - With one of two windows clicked in, use keyboard shortcuts to control video acquisition. First, press [T] to preview foreground image to optimize the webcam setting (exposure, contrast, etc) and lighting. Press [T] again to come back to raw frame view. Once nice & stable fish foreground is obtained, press [R] to start the recording and [R] again to stop. You can set a fix amount of time or frames to record in the GUI as well. [ECS] to abort from the viewer.


* Tracking (multitrack.py)

   This is for analysing the video recorded with TopSideMonitor_FrontEnd.py. It tracks fish with a simple background subtraction method (opencv MOG) and extracts from 3D fish trajectory various parameters to characterize feeding behavior.
     
     Tracking
       1. Open a video file from Menu File -> Open or just drag and drop it.
       2. Register a fish name (eg. GCaMP6_fish01) and [Register fish/Save] and choose it from the pull down menu on right. You can use [MS222] button to remove the fish.
       3. Set ROI for TopView by adjusting parameters starting with TV (Top View). Do this for Side View as well.
       4. If more than one fish is in the video, you can repeat 2 and 3. to regisrer more.
       5. Press [Play/Track] to start tracking.
       6. Press [Register fish/Save] when tracking is done.

      Correcting and preparing dnn training data

       1. Use pull-down menu on lower left to switch from "Track online" to "Replay mode".
       2. Press [Correct x,y] button to enable correction mode
       3. Hold Ctrl key and click the head position.
       4. Click the chest position (without Ctrl). These two positions define the fish orientation.
       5. Clicking the chest position will advance the frame at the step size defined in "Frame step" spin contol. The frames that both head and chest positions are manually entered will be marked for dnn training. Doing Online tracking for these frames will remove them from training data.
       6. When creating dnn training data, choose a large "Frame step" (e.g. 250 for 30 fps) in spin control to reduce similarity in training frames.

    Analyzing olfactory conditioning behavior

       1. Register events (e.g CS+, US) either using the GUI (not recommended) or by preparing an excel sheet. First collumn is fish name, 2nd is event label, 3rd is the event frame number (refer to example_event_sheet.xlsx) and drag and drop an excel file will overwrite the event meta data.
       2. Menu Alysis -> Create PDF report to get a PDF summary of this analysis and npz file containing tracking data and analysis results.

* Deep neural net models for head and chest detection (models\lasagne\fishmodel_****.py)

    When the built-in estimator of head and chest positions in TopSideMonitor does not work well enough, you can create train a deep neural net for your video to improve detection. 
    Under models\lasagne folder, put a .py file with the file name starting with "fishmodel_" prefix in which you can define a neural network for tracking. Take a look at the example model "fishmodel_nouriNet6.py" for details. The example pre-trained model included in TopSideMonitor may not perform well on your data. It would therefore be important to train your model with sufficient amount of your data. Once the model is loaded into multitrack.py GUI environment, creating training data and training the model can be done in the GUI. 


License
=======

TopSideMonitor is licensed under a 3-clause BSD style license - see the LICENSE.txt file.
