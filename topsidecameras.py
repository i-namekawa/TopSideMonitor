import datetime, os, subprocess, sys
from time import time, sleep

import cv2
import numpy as np

cwd = os.getcwd()
import motmot.cam_iface.cam_iface_ctypes as cam_iface
if not cam_iface.get_num_cameras():
    print 'No IEEE1394 camera found'
    cam_iface = None
os.chdir(cwd)


def twocameras(camerafps, videofps, datafolder, maxTime, qscale, textOn, textcolor, textpos, parent):
    
    # top view camera
    if cam_iface: # initilize IEEE1394 camera when available
        '''
        IEEE1394 camera is controlled by a patched motmot.cam_iface-0.5.4 module 
        on Python 2.7 and fview-flytrax-0.6.5 (cam_iface_mega.dll) 
        from the motmot project (http://code.astraw.com/projects/motmot/).
        
        For this patch, egg is deleted and source files in build\lib\motmot\cam_iface 
        are manually copied and __init__.py added to make it a python package.
        
        add unicode to this line in cam_iface_ctypes.py 
        line36: backend_path = unicode(os.environ.get('CAM_IFACE_CTYPES_PATH',None))

        Then, set the enviromental variable "CAM_IFACE_CTYPES_PATH" to 
        r'C:\Program Files (x86)\fview-flytrax-0.6.5'.
        '''
        mode_num = 0
        device_num = 0
        num_buffers = 32
        cam = cam_iface.Camera(device_num, num_buffers, mode_num)
        cam.start_camera()
        frame = cam.grab_next_frame_blocking()
        height, width = frame.shape
    else:
        cam = cv2.VideoCapture(-1) # -1: if there more than one, a dialog to choose will pop up
        success, frame = cam.read()
        if success:
            height, width, _ = frame.shape
            print height, width
        else:
            print 'top webcam failed to initialize'
    
    # side view camera
    webcam = cv2.VideoCapture(-1)  # -1: choose a 2nd camera from dialog
    success, frame = webcam.read()
    if success:
        height, width, _ = frame.shape
        print height, width
    else:
        print 'side webcam failed to initialize'
    webcam.set(cv2.cv.CV_CAP_PROP_FPS, camerafps)
    

    # initilize parameters
    recording = False
    tracking = False
    fno = 0

    if parent:
        parent.gauge.SetValue(0)
    _x = 1280-width-15 # hardcoding x position
    cv2.namedWindow('TopView')
    cv2.moveWindow('TopView', x=_x, y=30)
    cv2.namedWindow('SideView')
    cv2.moveWindow('SideView', x=_x, y=1024-height+52)

    while True:

        if cam_iface:
            frame1 = cam.grab_next_frame_blocking()
        else:
            success, frame1 = cam.read()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        success, frame2 = webcam.read()
        
        if success:

            
            if recording:
                
                elapsed = datetime.timedelta(seconds=time() - t0)
                if textOn:
                    cv2.putText(frame1,
                                'elapsed %s, frame %d'%(str(elapsed)[:-4], fno),
                                textpos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, textcolor)

                buf = np.vstack( (frame1, cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)) )
                p.stdin.write( buf.tostring() ) # faster than PIL
                
                if maxTime < time()-t0 and maxTime > 0:
                    p.stdin.close()
                    recording = False
                    fno = 0
                    
                    if parent:
                        parent.datafolder.SetBackgroundColour((255,255,255,255))
                        parent.gauge.SetValue(0)
                        parent.LED.SetValue('0')
                        parent.Refresh()

                    sleep(1)
                    if maxTime:
                        print '\n%d s of recording done.\nFile: %s\n' % (maxTime, fp)
                
                elif parent:
                    parent.LED.SetValue( "%02d %02d %02d" % (elapsed.seconds // 3600, 
                                                        elapsed.seconds//60 % 60, 
                                                        elapsed.seconds % 60 ))
                    #parent.gauge.SetValue(int((fno+1)/(maxFrame-1)*100))
                    if maxFrame:
                        parent.gauge.SetValue((fno+1+1)/(maxFrame-1)*100)
                
                fno +=1
            
            if tracking:
                mog1.apply(frame1.copy(), fgmask1, -3)
                mog2.apply(frame2.copy(), fgmask2, -3)
                
                #cv2.updateMotionHistory(fgmask1, mhi1, time()-t0, 3)
                #cv2.updateMotionHistory(fgmask2, mhi2, time()-t0, 3)
                #cv2.imshow("TopView", mhi1)
                #cv2.imshow("SideView", mhi2)
                
                cv2.imshow("TopView", fgmask1)
                cv2.imshow("SideView", fgmask2)
                
            else:
                cv2.line(frame1, pt1=(0,240), pt2=(640,240),color=155)
                cv2.line(frame1, pt1=(610,0), pt2=(610,480),color=155)
                cv2.imshow("TopView", frame1)
                
                # vertical line
                cv2.line(frame2, pt1=(320,0), pt2=(320,480),color=(55,255,255))
                
                # horizontal lines
                cv2.line(frame2, pt1=(0,45), pt2=(640,45),color=(55,255,255))
                cv2.line(frame2, pt1=(0,225), pt2=(640,225),color=(55,255,255))
                # inflow tubes
                cv2.line(frame2, pt1=(40,66), pt2=(46,111),color=(55,255,255))
                cv2.line(frame2, pt1=(600,66), pt2=(594,111),color=(55,255,255))
                
                cv2.imshow("SideView", frame2)
        
        
        char = cv2.waitKey(1) # need this to update windows
        if char == 27: # ESC
            if parent:
                if not parent.lock.GetValue():
                    break
            else:
                break
        
        elif char == 116: # "T"
            
            if not tracking and not recording:
                mog1 = cv2.BackgroundSubtractorMOG(
                    history = 30, 
                    nmixtures = 3,          # normally 3-5
                    backgroundRatio = 0.1,  # normally 0.1-0.9
                    noiseSigma = 15         # nomally 15 
                )
                fgmask1 = np.zeros((height, width), dtype=np.uint8)
                mhi1 = np.zeros((height, width), dtype=np.float32)

                mog2 = cv2.BackgroundSubtractorMOG(
                    history = 30, 
                    nmixtures = 3,          # normally 3-5
                    backgroundRatio = 0.1,  # normally 0.1-0.9
                    noiseSigma = 15         # nomally 15 
                )
                height2, width2, colordepth = frame2.shape
                fgmask2 = np.zeros((height2, width2), dtype=np.uint8)
                mhi2 = np.zeros((height2, width2), dtype=np.float32)
                t0 = time()
                tracking = True
            elif not recording:
                tracking = False
            

        elif char == 114: # "R"

            if recording:
                
                if parent:
                    if not parent.lock.GetValue():
                        recording = False
                        p.stdin.close()
                        fno = 0
                else:
                    recording = False
                    p.stdin.close()
                    fno = 0
                
                if parent:
                    parent.datafolder.SetBackgroundColour((255,255,255,255))
                    parent.gauge.SetValue(0)
                    parent.LED.SetValue('0')
                    parent.Refresh()
            else:
                
                if parent:
                    videofps = parent.videofps.GetValue()
                    camerafps = parent.camerafps.GetValue()
                    textpos = tuple([int(aa) for aa in parent.textpos.GetValue().split(',')])
                    textcolor = int(parent.textcolor.GetValue())
                    textOn = parent.textOn.GetValue()
                    qscale = parent.qscale.GetValue()
                    if parent.freeRun.GetValue():
                        maxTime = None
                        maxFrame = 0
                    else:
                        maxFrame = parent.maxFrame.GetValue()
                        maxTime = maxFrame / camerafps
                        #maxFrame = float(maxTime * camerafps)
                    print 'maxFrame', maxFrame
                
                recording = True
                fname = datetime.datetime.today().strftime( "%b-%d-%Y_%H_%M_%S.avi" )
                fp = os.path.join(datafolder, fname)
                cmdstring = ['.\\resources\\ffmpeg.exe',
                                '-y',
                                '-r', '%f' % videofps, 
                                #'-s', 'vga',
                                '-s', '%d, %d' % (width, height*2),
                                '-an',
                                '-analyzeduration', '0',  # skip auto codec analysis
                                
                                '-vf', 'scale=0',
                                '-f', 'rawvideo',
                                '-pix_fmt', 'gray', 
                                '-vcodec', 'rawvideo',
                                '-i', '-',
                                #'-pix_fmt','yuv420p',
                                '-vcodec', 'mpeg4']  # should be same as libxvid
                if qscale>31:
                    cmdstring.append('-b')
                else:
                    cmdstring.append('-qscale')
                cmdstring.extend([str(qscale),fp])
                
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                p = subprocess.Popen(cmdstring, 
                                    stdin=subprocess.PIPE,
                                    bufsize=-1,
                                    startupinfo=startupinfo
                                    )
                t0 = time()
                if parent:
                    parent.datafolder.SetBackgroundColour((255,255,100, 155))
                    parent.Refresh()


    # before exiting process
    if cam_iface:
        cam.close()
    else:
        cam.release()
    webcam.release()
    cv2.destroyAllWindows()
    if parent:
        parent.gauge.SetValue(0)
        parent.LED.SetValue('0')
        parent.Refresh()


if __name__ == '__main__':
    
    camera = twocameras( 
        camerafps=30, 
        videofps=10, 
        datafolder=r'D:\Data\itoiori\zebralab\2015\2015-02-16\test', 
        maxTime=5, 
        qscale=5,
        textOn=False,
        textcolor=222, 
        textpos=(100,220),
        parent=None
        )
