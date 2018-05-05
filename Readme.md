TopSideMonitor: fish behavior recording and tracking
====================================================

TopSideMonitor is a python program for recording and anylyzing adult zebrafish behavior developed for our study ["Rapid olfactory discrimination learning in adult zebrafish"](https://www.biorxiv.org/content/early/2018/05/04/314849). TopSideMonitor uses two orthogonal webcams to track fish in 3D (Top view and Side view) and extract various parameters (xyz trajectory, swimming speed, water surface sampling, distance to a point of interest etc) useful to characterize fish behaviors.

#Installation

Tested on Python 2.7 series on Windows 7. 

Install dependencies: numpy, scipy, wxpython, opencv, matplotlib, xlrd, xlwt.

This can be done by miniconda command below:

```conda create -n topside python=2.7 matplotlib scipy scipy wxpython xlrd xlwt```

Copy some reasonably new ffmpeg.exe binary into resourse folder.

(Optional)
For a FireWire camera, install motmot.cam_iface from http://code.astraw.com/projects/motmot/.




#How to use

* TopSideMonitor_FrontEnd.py

  - This is the GUI frontend for recording fish behavior. Run this script to set the recording parameters in GUI and press [Launch] button to open two camera frame viewers.
   
  - With one of two frame viewers clicked, use keyboard shortcuts to control video acquisition. First, press [T] to preview foreground image to optimize the webcam setting (exposure, contrast, etc) and lighting. Once nice & stable fish foreground is obtained, press [T] again to come back to raw frame view. Now, press [R] to start the recording and [R] again to stop. You can set a fix amount of time or frames to record in the GUI as well. [ECS] will close and exit from the viewer.


* multitrack.py

   This is for analysing the video generated with TopSideMonitor_FrontEnd.py. It tracks fish with a simple background subtraction method (opencv MOG) and extracts from 3D fish trajectory various parameters to characterize feeding behavior.
     
     Tracking
       1. Open a video file from Menu File -> Open or just drag and drop it.
       2. Register a fish name (eg. GCaMP6_fish01) and [Register fish/Save] and choose it from the pull down menu on right. You can use [MS222] button to remove the wrong entry.
       3. Set ROI for TopView by adjusting parameters starting with TV (Top View). Repeat this for Side View.
       4. If more than one fish is in the video, you can repeat 2 and 3 to regisrer more.
       5. Press [Play/Track] to start tracking.
       6. Press [Register fish/Save] when tracking is done.

    Analyzing olfactory conditioning behavior

       1. Register events either using the GUI (not recommended) or by preparing an excel sheet. First collumn is fish name, 2nd is event label, 3rd is the event frame number (refer to example_event_sheet.xlsx) and drag and drop an excel file will overwrite the events.
       2. Menu Analysis -> Create PDF report to get a PDF summary of this analysis and npz file containing tracking data and analysis results.


License
=======

TopSideMonitor is licensed under a 3-clause BSD style license - see the LICENSE.txt file.
