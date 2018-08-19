TopSideMonitor: fish behavior recording and tracking
====================================================

TopSideMonitor is a python program for recording and analyzing adult zebrafish behavior developed for our study ["Rapid olfactory discrimination learning in adult zebrafish" Namekawa et al (2018) Experimental Brain Research](https://rdcu.be/4bM2). TopSideMonitor uses two orthogonal webcams (Top and Side views) to track fish in 3D and extract various parameters (xyz trajectory, swimming speed, water surface sampling, distance to a point of interest etc) useful to characterize fish behaviors.

![Tracking](https://github.com/i-namekawa/TopSideMonitor/blob/master/resources/tracking.jpg)

`TopSideMonitor_FrontEnd.py` stacks top and side views in one video and `multitrack.py` can track multiple fish in one video.

# Installation

For Windows 7 users, binary files are available from the `release tab` at github. For Window 10, follow the instructions below.


Install dependencies: numpy (v1.9.2), scipy (v0.15.1), wxpython (v2.8.12), opencv (v2.4.12), matplotlib (v1.4.3), xlrd, xlwt. These specific library versions are all important.

The most libraries can be installed by miniconda commands below:

```
conda create -n topside27 python=2.7.10

conda activate topside27

conda install numpy=1.9.2 scipy=0.15.1 matplotlib=1.4.3

conda install -c krisvanneste wxpython

conda install xlrd xlwt
```

For opencv, download the official binary from https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.12/opencv-2.4.12.exe/download

Extract and copy `cv2.pyd` in `opencv\build\python\2.7\x64` to `site-packages` of your conda environment (e.g. `C:\Users\your_user_name\AppData\Local\Continuum\miniconda3\envs\topside27\Lib\site-packages`).

Copy everything (all starting with `opencv_`) in `opencv\build\x64\vc12` folder to the `site-packages`.

Copy `opencv_ffmpeg2412_64.dll` in `opencv\build\x64\vc12` next to your conda `python.exe` (i.e., `miniconda3\envs\topside27` folder, two levels above `site-packages`).

(Optional)
When using a FireWire camera, install `motmot.cam_iface` from http://code.astraw.com/projects/motmot/.


# How to use

Once installed, activate the conda environment by `conda activate topside27` from Anaconda prompt and then start the script by `python TopSideMonitor_FrontEnd.py` for recording and by `python multitrack.py` for analysis.

![Recording FrontEnd](https://github.com/i-namekawa/TopSideMonitor/blob/master/resources/recGUI.jpg)

* TopSideMonitor_FrontEnd.py

  - This is the GUI frontend for recording fish behavior. Run this script to set the recording parameters in GUI and press [Launch] button to open two camera frame viewers.
   
  - With one of two frame viewers clicked, use keyboard shortcuts to control video acquisition. First, press `T` to preview foreground image to optimize the webcam setting (exposure, contrast, etc) and lighting. Once nice & stable fish foreground is obtained, press `T` again to come back to raw frame view. Now, press `R` to start the recording and `R` again to stop. You can set a fix amount of time or frames to record in the GUI as well. `ECS` will close and exit from the viewer.

![Tracking tool](https://github.com/i-namekawa/TopSideMonitor/blob/master/resources/multitrackGUI.jpg)

* multitrack.py

  This is for analysing the video generated with TopSideMonitor_FrontEnd.py. It tracks fish with a simple background subtraction method (opencv MOG) and extracts from 3D fish trajectory various parameters to characterize feeding behavior.
     
  Tracking

    1. Open a video file from Menu File -> Open or just drag and drop it.
    2. Register a fish name (eg. Fish01) and `Register fish/Save` and choose it from the pull down menu on right. You can use `Remove` button to remove the wrong entry.
    3. Set ROI for TopView by adjusting parameters starting with TV (Topleft x1,y1; Bottomright x2,y2). 
    4. Set 6 parameters for Side View starting with SV (Topleft x1,y1; Bottomright x2,y2; outside corner of the other wall x3,y3).
    5. Set TV/SV noise blob size in pixels. Anything below this size will be discarded as noise blobs.
    6. Clock on `InflowTubes` and click a point inside Top View ROI and a point inside Side View ROI. This defines a point of interest to measure distance from fish.
    7. Click on `TopView ring` or `SideView ring` and draw a polygon around the feeding circle (8 points).
    8. (Option) TV ROI Head can rotate the Top View ROI if needed.
    9. If more than one fish is in the video, you can repeat 2 and 8 to regisrer more.
    10. Press `Play/Track` to start tracking.
    11. Press `Register fish/Save` when tracking is done.

  Analyzing olfactory conditioning behavior

    After the tracking is done for the portion of video that you are interested in, you can register events and obtain various behavior parameters around events.

    1. Register events either using the GUI (not recommended) or by preparing an excel sheet. First collumn is fish name, 2nd is event label, 3rd is the event frame number (refer to example_event_sheet.xlsx) and drag/drop the excel file on GUI will overwrite the events.
    2. Menu Analysis -> Create PDF report to get a PDF summary of this analysis and npz/mat file containing tracking data and analysis results.


License
=======

TopSideMonitor is licensed under a 3-clause BSD style license - see the LICENSE.txt file.
