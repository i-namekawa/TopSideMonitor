TopSideMonitor: fish behavior recording and tracking
====================================================

TopSideMonitor is a python program used for recording and anylysis of zebrafish olfactory conditioning behavior (a paper in preparation). TopSideMonitor uses two webcams to track fish in 3D (Top view and Side view).

#Installation

Tested on Python 2.7 series on Windows 7.

Install numpy, scipy, wxpython, opencv, matplotlib, xlrd, xlwt.

Copy some reasonably new ffmpeg.exe binary into resourse folder.

(Optional)
For a FireWire camera, install motmot.cam_iface from http://code.astraw.com/projects/motmot/.


#How to use

* TopSideMonitor_FrontEnd.py

  - This is the GUI frontend for recording fish behavior. Run this script to get the GUI and set parameters and press [Launch] button to open two camera frame viewers.
   
  - With one of two windows clicked in, use keyboard shortcuts to control video acquisition. First, press [T] to preview foreground image to optimize the webcam setting (exposure, contrast, etc) and lighting. Press [T] again to come back to raw frame view. Once nice & stable fish foreground is obtained, press [R] to start the recording and [R] again to stop. You can set a fix amount of time or frames to record in the GUI as well. [ECS] to abort from the viewer.


* multitrack.py

   This is for analysing the video recorded with TopSideMonitor_FrontEnd.py.
     
     Tracking
       1. Open a video file from Menu File -> Open or just drag and drop it.
       2. Register a fish name (eg. GCaMP6_fish01) and [Register fish/Save] and choose it from the pull down menu on right. You can use [MS222] button to remove the fish.
       3. Set ROI for TopView by adjusting parameters starting with TV (Top View). Do this for Side View as well.
       4. If more than one fish is in the video, you can repeat 2-4. to regisrer more.
       5. Press [Play/Track] to start tracking.
       6. Press [Register fish/Save] when tracking is done.

    Analyzing olfactory conditioning behavior

       1. Register events either using the GUI (not recommended) or by preparing an excel sheet. First collumn is fish name, 2nd is event label, 3rd is the event frame number (refer to example_event_sheet.xlsx) and drag and drop an excel file will overwrite the events.
       2. Menu Alysis -> Create PDF report to get a PDF summary of this analysis and npz file containing tracking data and analysis results.
