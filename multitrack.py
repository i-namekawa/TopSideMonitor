from __future__ import division
import os, sys, subprocess
from glob import glob
import cPickle as pickle

import cv2
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import xlrd

import wx
import wx.lib.agw.floatspin as FS
import wx.lib.agw.multidirdialog as MDD

from plotting import *

# TWOCHOICE = True
TWOCHOICE = False


# TODO: 
# not woking with cv2 3.1.0-dev. 


def interporatePolyOverFrames(ringpolyDict, nmax_frame):
    Ringpolyarray = np.ones((nmax_frame, 5)) * np.nan
    for n, poly in ringpolyDict.items():
        (x,y), (w,h), angle = cv2.fitEllipse(np.array(poly))
        Ringpolyarray[n,:] = x, y, w, h, angle
    Ringpolyarray[:,0] = interp_nan(Ringpolyarray[:,0]).astype(int)
    Ringpolyarray[:,1] = interp_nan(Ringpolyarray[:,1]).astype(int)
    Ringpolyarray[:,2:] = np.nanmean( Ringpolyarray[:,2:], axis=0 ).astype(int)
    return Ringpolyarray

def interporatexyOverFrames(tubexyDict, nmax_frame):
    temp = np.ones((nmax_frame, 2)) * np.nan
    for n, (x,y) in tubexyDict.items():
        temp[n,:] = x, y
    temp[:,0] = interp_nan(temp[:,0]).astype(int)
    temp[:,1] = interp_nan(temp[:,1]).astype(int)
    return temp.astype(int)

def add_removePickledDict(fp, data, key, remove=False):
    with open(fp, 'rb') as f:
        temp = pickle.load(f)
    with open(fp, 'wb') as f:
        if remove and temp.has_key(key):
            temp.pop(key)
        else:
            temp[key] = data
        pickle.dump(temp, f)

class RedirectText(object):  # from a blog: www.blog.pythonlibrary.org
    def __init__(self, aWxTextCtrl):
        self.out=aWxTextCtrl
    def write(self,string):
        self.out.SetInsertionPointEnd()  # go to the bottom
        self.out.WriteText(string)

class FileDrop(wx.FileDropTarget):
    def __init__(self, parent):
        wx.FileDropTarget.__init__(self)
        self.parent = parent
    def OnDropFiles(self, x, y, filenames):
        if filenames[0].endswith('.avi'):
            self.parent.LoadAvi(None, filenames[0])
            self.parent.OnnamedWinbtn(None)
            return
        elif filenames[0].endswith('.xlsx'):
            self.parent.LoadExcel(filenames[0])
        elif filenames[0].endswith('.pickle'):
            # loading previous fish names
            self.parent.choiselist = self.parent.getFishnamesfromPickle(None, filenames[0])
            self.parent.OnChoice(events=None, settingfp=filenames[0])
            self.parent.OnMogParams(None, 'temp') # init self.mog with the params in GUI
            self.parent.RefreshChoise(self.parent.targetfish, self.parent.choiselist)
            self.parent.updateCmap()
            self.parent.OnSpin(None)


class wxGui(wx.Frame):
    def __init__(self, pos):
        wx.Frame.__init__(self, None, pos=pos, title='Parameter interface')
        self.SetDropTarget(FileDrop(self))
        self.SetIcon(fishicon)

        self.fp = None
        self.cap = None
        self.curframe = 0
        self.namedWindow = False
        self.clicks = []

        # Build menu bar
        menuBar = wx.MenuBar()
        
        FileMenu = wx.Menu()
        FileMenu.Append(101, "&Open", "Open an avi file")
        FileMenu.Append(102, "Open &data folder")
        FileMenu.Append(103, "&Quit", "Quite the application")
        FileMenu.Append(104, "Record video")
        self.Bind(wx.EVT_MENU, self.OnOpen, id=101)
        self.Bind(wx.EVT_MENU, self.OnContainingFolder, id=102)
        self.Bind(wx.EVT_MENU, self.OnQuit, id=103)
        self.Bind(wx.EVT_MENU, self.OnRec, id=104)

        AnalysisMenu = wx.Menu()
        AnalysisMenu.Append(202, "Quick plot")
        self.Bind(wx.EVT_MENU, self.OnQuickPlot, id=202)
        self.eachReport = AnalysisMenu.Append(201, "Create a PDF report for selected fish")
        self.Bind(wx.EVT_MENU, self.OngetPDFs, id=201)
        AnalysisMenu.Append(203, "Summary PDF report from multiple folders")
        self.Bind(wx.EVT_MENU, self.OnSummaryPDF, id=203)

        
        self.OptionMenu = wx.Menu()
        self.OptionMenu.Append(301, "Copy current event data to all other fish")
        self.Bind(wx.EVT_MENU, self.OnCopyEvents, id=301)
        self.OptionMenu.Append(302, "Save and quit when tracking done", "Check Item", wx.ITEM_CHECK)
        self.needMatfile = self.OptionMenu.Append(303, "Export mat file as well", 'MATLAB export', wx.ITEM_CHECK)
        self.connectSplitBlobs = self.OptionMenu.Append(304, "Connect split fish under ring", "Check Item", wx.ITEM_CHECK)
        self.connectSplitBlobs.Check(True)

        self.OptionMenu.Append(305, "low pass filter headx, heady around here")
        self.Bind(wx.EVT_MENU, self.Onfilterheadxy, id=305)

        menuBar.Append(FileMenu, "&File")
        menuBar.Append(AnalysisMenu, "&Analysis")
        menuBar.Append(self.OptionMenu, "&Options")
        self.SetMenuBar(menuBar)

        # File history
        self.filehistory = wx.FileHistory()
        self.config = wx.Config('MultiFishTrack', style=wx.CONFIG_USE_LOCAL_FILE)
        self.filehistory.Load(self.config)
        self.filehistory.UseMenu(FileMenu)
        self.filehistory.AddFilesToMenu()
        self.Bind(wx.EVT_MENU_RANGE, self.OnFileHistory, id=wx.ID_FILE1, id2=wx.ID_FILE9)

        # status bar
        self.sb = self.CreateStatusBar()

        # btn etc
        self.fishname = wx.SearchCtrl(self, size=(90,25))
        self.fishname.SetDescriptiveText('Type in fish name here and register')
        self.fishname.ShowSearchButton(False)
        self.savebtn = wx.Button(self, label='Register fish/Save', style=wx.BU_EXACTFIT)
        self.removebtn = wx.Button(self, label='MS222', style=wx.BU_EXACTFIT)
        self.choiselist = []
        self.targetfish = wx.Choice(self, choices = self.choiselist)
        
        self.depth = FS.FloatSpin(self, min_val=0.1, max_val=10.0, 
                                    increment=0.01, value=1.00, style=FS.FS_LEFT)
        self.depth.SetDigits(2)

        self.namedWinbtn = wx.Button(self, label='Open OpenCV', style=wx.BU_EXACTFIT)
        self.namedWinbtn.Disable()

        self.fgmaskbtn = wx.Button(self, label='Show fgmask', style=wx.BU_EXACTFIT)
        self.showfgmask = False

        self.ringAppearochLevel = wx.SpinCtrl(self, size=(85,-1), min=-1, max=9999, initial=62)

        self.groundLevel = wx.SpinCtrl(self, size=(85,-1), min=-1, max=9999, initial=960)

        self.ROITVx1 = wx.SpinCtrl(self, size=(85,-1), min=0, max=640, initial=3)
        self.ROITVy1 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=10)
        self.ROITVx2 = wx.SpinCtrl(self, size=(85,-1), min=0, max=640, initial=612)
        self.ROITVy2 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=225)
        self.ROITVH  = wx.SpinCtrl(self, size=(85,-1), min=-180, initial=0)

        self.inflowTubebtn = wx.ToggleButton(self, label='InflowTubes', style=wx.BU_EXACTFIT)
        self.inflowTubebtn.Bind(wx.EVT_TOGGLEBUTTON, self.OninflowTube)

        self.ROISVx1 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=318)
        self.ROISVy1 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=585)
        self.ROISVx2 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=636)
        self.ROISVy2 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=774)
        self.ROISVx3 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=500)
        self.ROISVy3 = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=700)

        self.ID_ringpolyTV = wx.NewId()
        self.ID_ringpolySV = wx.NewId()
        self.ringpolyTVbtn = wx.Button(self, self.ID_ringpolyTV, label='TopView ring', style=wx.BU_EXACTFIT)
        self.ringpolySVbtn = wx.Button(self, self.ID_ringpolySV, label='SideView ring', style=wx.BU_EXACTFIT)
        self.ringpolyTVbtn.Disable()
        self.ringpolySVbtn.Disable()
        self.ringpolyTV = False # To emurate toggle btn
        self.ringpolySV = False

        self.TVnoiseSize, self.SVnoiseSize = 150, 150
        self.noise_blob_sizeTV = wx.SpinCtrl(self, size=(85,-1), min=0, max=9999, initial=self.TVnoiseSize)
        self.noise_blob_sizeSV = wx.SpinCtrl(self, size=(85,-1), min=0, max=9999, initial=self.SVnoiseSize)

        self.mog_history = wx.SpinCtrl(self, size=(85,-1), min=0, max=9999, initial=25)
        self.mog_nmixtures = wx.SpinCtrl(self, size=(85,-1), min=0, max=9, initial=3)
        self.mog_backgroundRatio = FS.FloatSpin(self, min_val=0.0, max_val=1.0,
                                    increment=0.01, value=0.05, style=FS.FS_LEFT)
        self.mog_backgroundRatio.SetDigits(2)
        self.mog_noiseSigma = wx.SpinCtrl(self, size=(85,-1), min=0, max=960, initial=12)
        self.mog_learning_rate = wx.SpinCtrl(self, size=(85,-1), min=-99, max=99, initial=-1)

        self.event_label = wx.SearchCtrl(self, size=(90,25))
        self.event_label.SetDescriptiveText('Define label here')
        self.event_label.ShowSearchButton(False)

        self.event_label_list = []
        self.event_label_choice = wx.Choice(self, size=(90,25), choices = self.event_label_list)

        self.register_event = wx.Button(self, label='Add/remove', style=wx.BU_EXACTFIT)
        self.preRange = wx.SpinCtrl(self, size=(85,-1), min=0, max=999999, initial=3600)

        self.Replay_Online = wx.Choice(self, choices=['Track online', 'Replay mode'])
        self.Replay_Online.SetSelection(0)
        self.TrackOnline = True

        self.correctbtn = wx.ToggleButton(self, label='Correct x,y', style=wx.BU_EXACTFIT)
        self.correctbtn.Disable()

        self.resetTracking = wx.Button(self, label='Reset tracking')
        self.resetTracking.Disable()
        self.resetTracking.SetBackgroundColour('yellow')

        self.trackingmode = wx.Choice(self, choices=['use both', 'force mog', 'force bg'])
        self.trackingmode.SetSelection(1)

        self.frameStepSpin = wx.SpinCtrl(self, size=(85,-1), initial=1, max=9999)

        self.playPausebtn = wx.Button(self, label='Play/Track', style=wx.BU_EXACTFIT)
        self.playPausebtn.Disable()
        self.playing = False

        self.slider = wx.Slider(self, style=wx.SL_HORIZONTAL, value=0, minValue=0, maxValue=100)
        self.sliderSpin = wx.SpinCtrl(self, size=(73,-1), initial=0)
        self.slider.Disable()
        self.sliderSpin.Disable()

        self.log = wx.TextCtrl(self, size=(372, 100),
            style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.EXPAND | wx.TE_AUTO_URL | wx.TE_READONLY)
        sys.stdout = RedirectText(self.log)
        self.log.SetFocus()

        # sizer
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.fishname, 1, wx.ALL|wx.EXPAND, 5)
        hbox.Add(self.savebtn, 0, wx.ALL|wx.EXPAND, 5)
        hbox.Add(self.removebtn, 0, wx.ALL|wx.EXPAND, 5)
        hbox.Add(self.targetfish, 0, wx.ALL|wx.EXPAND, 5)

        gbs = wx.GridBagSizer(8,5)

        gbs.Add(self.namedWinbtn, (0,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.fgmaskbtn, (0,1), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='Depth correction'), (1,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.depth, (1,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='ringApproachLevel'), (1,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ringAppearochLevel, (1,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='Crop video (y)'), (1,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.groundLevel, (1,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='TV ROI x1,y1'), (2,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROITVx1, (2,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROITVy1, (2,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='SV ROI x1,y1'), (2,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROISVx1, (2,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROISVy1, (2,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='TV ROI x2,y2'), (3,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROITVx2, (3,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROITVy2, (3,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='SV ROI x2,y2'), (3,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROISVx2, (3,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROISVy2, (3,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='TV ROI Head'), (4,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROITVH, (4,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.inflowTubebtn, (4,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='SV ROI x3,y3'), (4,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROISVx3, (4,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ROISVy3, (4,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='TV noise blob size'), (5,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.noise_blob_sizeTV, (5,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ringpolyTVbtn, (5,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='SV noise blob size'), (5,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.noise_blob_sizeSV, (5,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.ringpolySVbtn, (5,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='MOG h m b n l'), (6,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.mog_history, (6,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.mog_nmixtures, (6,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.mog_backgroundRatio, (6,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.mog_noiseSigma, (6,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.mog_learning_rate, (6,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(wx.StaticText(self, label='Register events:'), (7,0), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.event_label, (7,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.event_label_choice, (7,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.register_event, (7,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='Pre duration\n in frame'), (7,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.preRange, (7,5), flag=wx.ALIGN_CENTRE)

        gbs.Add(self.Replay_Online, (8,0), span=(1,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.correctbtn, (8,1), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.trackingmode, (8,2), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.resetTracking, (8,3), flag=wx.ALIGN_CENTRE)
        gbs.Add(wx.StaticText(self, label='Frame step:'), (8,4), flag=wx.ALIGN_CENTRE)
        gbs.Add(self.frameStepSpin, (8,5), flag=wx.ALIGN_CENTRE)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.playPausebtn, 0, wx.ALL|wx.EXPAND, 2)
        hbox2.Add(self.slider, 1, wx.ALL|wx.EXPAND, 2)
        hbox2.Add(self.sliderSpin, 0, wx.ALL|wx.EXPAND, 2)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(hbox, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 2)
        mainSizer.Add(gbs, 0, wx.ALL|wx.EXPAND, 2)
        mainSizer.Add(hbox2, 0, wx.ALL|wx.EXPAND, 2)
        mainSizer.Add(self.log, 1, wx.ALL|wx.EXPAND, 2)

        self.SetSizer(mainSizer)
        self.SetAutoLayout(True)
        self.Fit()

        # Bind events
        self.savebtn.Bind(wx.EVT_BUTTON, self.Onsavebtn)
        self.removebtn.Bind(wx.EVT_BUTTON, self.Onremovebtn)
        self.targetfish.Bind(wx.EVT_CHOICE, self.OnChoice)
        self.namedWinbtn.Bind(wx.EVT_BUTTON, self.OnnamedWinbtn)
        self.ringpolyTVbtn.Bind(wx.EVT_BUTTON, self.OnRingpolybtn)
        self.ringpolySVbtn.Bind(wx.EVT_BUTTON, self.OnRingpolybtn)

        self.resetTracking.Bind(wx.EVT_BUTTON, self.OnresetTracking)

        self.fgmaskbtn.Bind(wx.EVT_BUTTON, self.Onfgmaskbtm)

        self.slider.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSlider) # mouse drag, wheele
        self.slider.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.OnSlider) # <- -> keyboard
        self.sliderSpin.Bind(wx.EVT_SPINCTRL, self.OnSliderSpin)

        self.ROITVx1.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROITVy1.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROITVx2.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROITVy2.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROITVH .Bind(wx.EVT_SPINCTRL, self.OnSpin)

        self.ROISVx1.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROISVy1.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROISVx2.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROISVy2.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROISVx3.Bind(wx.EVT_SPINCTRL, self.OnSpin)
        self.ROISVy3.Bind(wx.EVT_SPINCTRL, self.OnSpin)

        self.groundLevel.Bind(wx.EVT_SPINCTRL, self.OnSpin)

        self.event_label_choice.Bind(wx.EVT_CHOICE, self.OnEventChoise)

        self.mog_history.Bind(wx.EVT_SPINCTRL, self.OnMogParams)
        self.mog_nmixtures.Bind(wx.EVT_SPINCTRL, self.OnMogParams)
        self.mog_backgroundRatio.Bind(FS.EVT_FLOATSPIN, self.OnMogParams)
        self.mog_noiseSigma.Bind(wx.EVT_SPINCTRL, self.OnMogParams)
        self.mog_learning_rate.Bind(wx.EVT_SPINCTRL, self.OnMogParams)
        self.noise_blob_sizeTV.Bind(wx.EVT_SPINCTRL, self.OnMogParams)
        self.noise_blob_sizeSV.Bind(wx.EVT_SPINCTRL, self.OnMogParams)

        self.register_event.Bind(wx.EVT_BUTTON, self.OnEventRegister)

        self.Replay_Online.Bind(wx.EVT_CHOICE, self.OnChoiceReplay_Track)

        self.playPausebtn.Bind(wx.EVT_BUTTON, self.Onplaybtn)

        self.timer = wx.Timer(self) # Timer for video
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
        self.timer.Start(1000/600.0) # aim at 600 fps!

        self.Show()

    def OninflowTube(self, event):
        fish = self.targetfish.GetStringSelection()
        print 'InflowTubeTVdict', self.InflowTubeTVdict[fish].items()
        print 'InflowTubeSVdict', self.InflowTubeSVdict[fish].items()

    def OnCopyEvents(self, event):
        fish = self.targetfish.GetStringSelection()
        if fish not in (u'', 'all'):
            registeredfish = [f for f in self.choiselist if f not in ('temp', 'mog', 'all')]
            print 'Copying event data for %s to all other fish' % fish
            
            for f in registeredfish:
                print 'Copy to fish', f, self.EventData[fish]
                self.loadsetting(f)
                self.savesetting(f, Copying=self.EventData[fish])
    
    def OnSummaryPDF(self, event):
        if self.fp:
            defaultPath = os.path.dirname(self.fp)
        else:
            defaultPath = ''
        dlg = MDD.MultiDirDialog(self, defaultPath=defaultPath)
        if dlg.ShowModal() == wx.ID_OK:
            folders = dlg.GetPaths()
            dlg.Destroy()
            print 'folders',folders


            pickle_files = []
            for eachfolder in folders:
                
                if eachfolder.find('(')>0: # MDD may return a path with volume label which glob fails
                    wovolmulabel = eachfolder[eachfolder.find('('):]
                    eachfolder = wovolmulabel.replace('(', '').replace(')', '')

                pattern = os.path.join(eachfolder, '*.pickle')
                
                for d in glob(pattern):
                    fname = os.path.basename(d)
                    print d, fname
                    if not fname.startswith('Summary') and not fname.endswith('- Copy.pickle'):
                        pickle_files.append(d)
            
            if pickle_files:
                dlg = wx.FileDialog( self, message="Save data file as ...", defaultDir=folders[0], 
                        defaultFile="Summary.pickle", wildcard="pickle (*.pickle)|*.pickle", style=wx.SAVE )
                if dlg.ShowModal() == wx.ID_OK:
                    _fp = dlg.GetPath()
                    data = getPDFs(pickle_files, createPDF=False)
                    with open(_fp, 'wb') as f:
                        pickle.dump(data, f)
                    dlg.Destroy()

                    print 'getSummary', os.path.dirname(_fp)
                    getSummary(data, os.path.dirname(_fp))
                    
                    if self.needMatfile.IsChecked():
                        pickle2mat(_fp, data)

                    dlg = wx.MessageDialog(self, 'done!', style=wx.OK)
                    if dlg.ShowModal() == wx.ID_OK:
                        dlg.Destroy()
            else:
                print 'no tracking data found'

    def OnRec(self, event):
        if self.fp:
            defaultPath = os.path.dirname(self.fp)
            dlg = wx.FileDialog( self, message="Save avi file as ...", defaultDir=defaultPath, 
                    defaultFile="clip.avi", wildcard="video (*.avi)|*.avi", style=wx.SAVE )
            if dlg.ShowModal() == wx.ID_OK:
                self.clipfp = dlg.GetPath()
                self.writer = cv2.VideoWriter( 
                                self.clipfp, 
                                # -1, # will open a dialog to choose codec
                                # cv2.cv.FOURCC('M','J','P','G'), 
                                cv2.cv.FOURCC('D','I','V','3'),  # 2260 kb 250 frames
                                fps=int(self.fps), 
                                frameSize = self.framesize[::-1], # opencv wants w,h
                                isColor=True
                                )
                dlg.Destroy()

    def OnQuickPlot(self, event):
        selected = self.targetfish.GetStringSelection()
        if selected not in ['', 'all']:

            tvx = self.TVtracking[selected][:,0]
            tvy = self.TVtracking[selected][:,1]
            svy = self.SVtracking[selected][:,1]
            headx = self.TVtracking[selected][:,3]
            heady = self.TVtracking[selected][:,4]
            x,y,z = map(interp_nan, [tvx,tvy,svy])
            
            ringpolyTVArray = self.ringpolyTVArray[selected]
            ringpolySVArray = self.ringpolySVArray[selected]
            ringAppearochLevel = self.ringAppearochLevel.GetValue()

            smoothedz, peaks_within = approachevents(x, y, z, 
                ringpolyTVArray, ringpolySVArray, thrs=ringAppearochLevel)
            # convert list to numpy array
            temp = np.zeros_like(x)
            temp[peaks_within] = 1
            peaks_within = temp
            
            preRange = self.preRange.GetValue()
            CS = self.curframe if self.curframe > preRange*1.2 else preRange*1.2
            USs = [CS+30*self.fps]
            events = CS, USs, preRange
            inflowpos = (self.InflowTubeTVArray[selected][:,0],
                        self.InflowTubeTVArray[selected][:,1],
                        self.InflowTubeSVArray[selected][:,1])
            
            TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = self.getSpinCtrls()
            swimdir, water_x, water_y = swimdir_analysis(x, y, z, ringpolyTVArray, ringpolySVArray,
                                            TVx1, TVy1, TVx2, TVy2, self.fps)
            # all of swimdir are within ROI   (frame#, inout, speed)
            sdir = np.array(swimdir)
            withinRing = sdir[:,1]>0 # inout>0 are inside polygon
            temp = np.zeros_like(x)
            temp[ sdir[withinRing,0].astype(int) ] = 1
            swimdir_within = temp
            
            heady, headx = map(interp_nan, [heady, headx])
            headx, heady = filterheadxy(headx, heady)
            dy = heady - y
            dx = headx - x
            _theta = np.arctan2(dy, dx)
            
            rng = np.arange(CS-preRange, CS+preRange, dtype=np.int)
            _theta[rng] = smoothRad(_theta[rng], thrs=np.pi/2) # this takes time. comipute only the part we need
            theta_shape = _theta
            dtheta_shape = np.append(0, np.diff(theta_shape))

            # velocity based
            cx, cy = filterheadxy(x, y)  # centroid x,y

            vx = np.append(0, np.diff(cx))
            vy = np.append(0, np.diff(cy))
            _theta = np.arctan2(vy, vx)

            _theta[rng] = smoothRad(_theta[rng], thrs=np.pi/2)
            theta_vel = _theta
            dtheta_vel = np.append(0, np.diff(theta_vel))

            kernel = np.ones(4)
            dthetasum_shape = np.convolve(dtheta_shape, kernel,'same')
            dthetasum_vel = np.convolve(dtheta_vel, kernel,'same')
            # 4 frames = 1000/30.0*4 = 133.3 ms 
            thrs = np.pi/4 * (133.33333333333334/120) # Braubach et al 2009 90 degree in 120 ms 

            peaks_shape = argrelextrema(abs(dthetasum_shape), np.greater)[0]
            turns_shape = peaks_shape[ (abs(dthetasum_shape[peaks_shape]) > thrs).nonzero()[0] ]

            peaks_vel = argrelextrema(abs(dthetasum_vel), np.greater)[0]
            turns_vel = peaks_vel[ (abs(dthetasum_vel[peaks_vel]) > thrs).nonzero()[0] ]

            _title='QuickPlot %s' % selected
            plot_eachTr(events, x, y, z, inflowpos, self.ringpixel[selected], peaks_within, 
                swimdir_within=swimdir_within, pp=None, _title=_title, fps=self.fps)
            plotTrajectory(x, y, z, events, fps=self.fps, pp=None, ringpolygon=None)
            plot_turnrates(events, dthetasum_shape, dthetasum_vel, turns_shape, turns_vel, pp=None, _title=_title)
            show()

    def OngetPDFs(self, event):
        selected = self.targetfish.GetStringSelection()
        if self.fp and selected:
            settingfp = self.fp[:-3]+'pickle'
            fname = os.path.basename(self.fp)
            
            if os.path.exists(settingfp):
                data = np.load(settingfp)

                # all fish found in data will be processed.
                if selected == 'all':
                    getPDFs([settingfp])
                # only single fish selected
                elif [k for k in self.EventData[selected].keys() if k.startswith('CS')]:
                    if (fname, selected) in data.keys():
                        self.savesetting(selected)
                        getPDFs([settingfp], [selected])
                    else:
                        return
                else:
                    return

                dlg = wx.MessageDialog(self, 
                    "pdf report for %s created.\nOpen the pdf?" % selected, style=wx.YES_NO)
                if dlg.ShowModal() == wx.ID_YES:
                    dlg.Destroy()

                    if selected == 'all':
                        registeredfish = [f for f in self.choiselist if f not in ('temp', 'mog', 'all')]
                    else:
                        registeredfish = [selected]
                    for fish in registeredfish:
                        pdfname = self.fp[:-4]+'_'+fish+'.pdf'
                        subprocess.Popen(pdfname, shell=True)

    def OnQuit(self, event):
        close('all')
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.Destroy()

    def OnOpen(self, events):
        dlg = wx.FileDialog(self, message="Open files...",
                defaultDir=os.getcwd(), #defaultFile=fp,
                wildcard='Avi files (*.avi)|*.avi',
                style=wx.OPEN | wx.CHANGE_DIR )
        if dlg.ShowModal() == wx.ID_OK:
            fp = dlg.GetPath()
            self.LoadAvi(None, fp)
        dlg.Destroy()
        self.OnnamedWinbtn(None)

    def OnFileHistory(self, events):
        fileNum = events.GetId() - wx.ID_FILE1
        fp = self.filehistory.GetHistoryFile(fileNum)
        self.LoadAvi(None, fp)
        self.OnnamedWinbtn(None)

    def OnContainingFolder(self, events):
        if self.fp:
            subprocess.call(r'explorer "%s"' % os.path.dirname(self.fp))

    def OnTimer(self, events):
        if self.playing:
            step = self.frameStepSpin.GetValue()
            self.curframe += step
            
            if self.curframe >= self.nmax_frame-1:
                self.curframe = 0
                self.Onplaybtn(None)
                # save data and quit
                fish = self.targetfish.GetStringSelection()
                if self.OptionMenu.IsChecked(id=302) and fish:
                    if fish == 'all':
                        registeredfish = [f for f in self.choiselist if f not in ('temp', 'mog', 'all')]
                        for f in registeredfish:
                            self.savesetting(f)
                    else:
                        self.savesetting(fish)
                    self.OnQuit(None)
            
            self.slider.SetValue(self.curframe)
            self.sliderSpin.SetValue(self.curframe)
            self.updateFrame()

    def Onfgmaskbtm(self, events):
        if self.showfgmask:
            self.showfgmask = False
            self.fgmaskbtn.SetBackgroundColour(wx.NullColour)
        else:
            self.showfgmask = True
            self.fgmaskbtn.SetBackgroundColour('gray')
        if self.namedWindow:
            self.updateFrame()

    def Onplaybtn(self, events):
        if self.playing:
            self.playing = False
            self.playPausebtn.SetBackgroundColour(wx.NullColour)
            self.playPausebtn.SetLabel('Play')
            if self.clipfp:
                print 'Video clip is generated.'
                self.writer.release()
                self.clipfp = False
            fish = self.targetfish.GetStringSelection()
            if fish not in ('', 'all'):
                st = 0 if self.curframe-100 < 0 else self.curframe-100
                n = self.curframe
                TVavg_area = int(np.nanmean( self.TVtracking[fish][st:n,2] ))
                SVavg_area = int(np.nanmean( self.SVtracking[fish][st:n,2] ))
                print ' TopView avg area %d,  SideView avg area %d (past 100 frames)' % (TVavg_area, SVavg_area)
                prerange = self.preRange.GetValue()
                x = interp_nan( self.TVtracking[fish][:,0] )
                y = interp_nan( self.TVtracking[fish][:,1] )
                z = interp_nan( self.SVtracking[fish][:,1] )
                speed3D = np.sqrt( np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2 )

                print 'Ring passing frames x4SD: ', [s for s in ( speed3D > speed3D.std()*4 ).nonzero()[0] if n<s<prerange+n]
                print '\tJumpy frames x6SD, std: ', [s for s in ( speed3D > speed3D.std()*6 ).nonzero()[0] if n<s<prerange+n], speed3D.std()

        else:
            self.playing = True
            self.playPausebtn.SetBackgroundColour('yellow')
            self.playPausebtn.SetLabel('Pause')

    def OnnamedWinbtn(self, events):
        if self.namedWindow:
            self.namedWindow = False
            self.namedWinbtn.SetLabel('Open OpenCV')
            cv2.destroyAllWindows()
        else:
            self.namedWindow = True
            self.namedWinbtn.SetLabel('Close OpenCV')
            cv2.namedWindow('Tracking')
            x,y,w,h = self.GetScreenRect().Get() # http://www.wxpython.org/docs/api/wx.Rect-class.html#Get
            cv2.moveWindow('Tracking', x=x+w, y=35)
            fish = self.targetfish.GetStringSelection()
            if not fish:
                self.TVtracking['temp'] = np.ones((self.nmax_frame,6)) * np.nan
                self.SVtracking['temp'] = np.ones((self.nmax_frame,6)) * np.nan
                self.ringpolyTVArray['temp'] = None
                self.ringpolySVArray['temp'] = None
                self.InflowTubeTVArray['temp'] = None
                self.InflowTubeSVArray['temp'] = None
                self.ringpixel['temp'] = np.ones(self.nmax_frame) * np.nan
            self.updateFrame()
            cv2.setMouseCallback('Tracking', self.onMouse)

    def OnSlider(self, event):
        n = self.slider.GetValue()
        self.sliderSpin.SetValue(n)
        self.curframe = n
        self.sb.SetStatusText('Time %02d:%02d' % (n/self.fps/60, n/self.fps%60))
        if self.namedWindow:
            self.updateFrame()

    def OnSpin(self, events):
        h,w = self.framesize
        GL = self.groundLevel.GetValue()
        n = self.curframe
        self.sb.SetStatusText('Time %02d:%02d' % (n/self.fps/60, n/self.fps%60))
        TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = self.getSpinCtrls()
        if GL != h:
            _ymax = max([TVy1, TVy2, SVy1, SVy2])
            GL = _ymax if GL < _ymax else GL  # if too small, clamp to _ymax
            self.groundLevel.SetValue(GL) # and apply
            self.framesize = GL, w
            self.fgmask = np.zeros(self.framesize, dtype=np.uint8)
        self.clipout = np.zeros(self.framesize, dtype=np.uint8)
        cx,cy = (TVx1+TVx2)/2, (TVy1+TVy2)/2
        boxTV = cv2.cv.BoxPoints(((cx,cy), (TVx2-TVx1, TVy2-TVy1), TVH))
        cv2.rectangle(self.clipout, (SVx1, SVy1), (SVx2, SVy2), 255, thickness=-1)
        if self.namedWindow and events:
            self.updateFrame()

    def getMogParams(self):
        history = self.mog_history.GetValue()
        nmixtures = self.mog_nmixtures.GetValue()
        backgroundRatio = self.mog_backgroundRatio.GetValue()
        noiseSigma = self.mog_noiseSigma.GetValue()
        learning_rate = self.mog_learning_rate.GetValue()
        return history, nmixtures, backgroundRatio, noiseSigma, learning_rate

    def OnMogParams(self, events, fish='temp'):
        history, nmixtures, backgroundRatio, noiseSigma, self.learning_rate = self.getMogParams()
        # Mixture of Gaussian
        self.mogTV[fish] = cv2.BackgroundSubtractorMOG(
            history = history,                  # normally 100
            nmixtures = nmixtures,              # normally 3-5
            backgroundRatio = backgroundRatio,  # normally 0.1-0.9
            noiseSigma = noiseSigma             # <7 starts to deteriorate, 15 is good
            )
        self.mogSV[fish] = cv2.BackgroundSubtractorMOG(
            history = history,
            nmixtures = nmixtures,
            backgroundRatio = backgroundRatio,
            noiseSigma = noiseSigma
            )
        print 'MOG settings: history=%d, nmixtures=%d, backgroundRatio=%1.2f, noiseSigma=%d, learning_rate=%d' % (
                    history, nmixtures, backgroundRatio, noiseSigma, self.learning_rate)

        TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = self.getSpinCtrls()
        self.fgmaskTV[fish] = np.zeros((TVy2-TVy1,TVx2-TVx1), dtype=np.uint8)
        self.fgmaskSV[fish] = np.zeros((SVy2-SVy1,SVx2-SVx1), dtype=np.uint8)

        # train the mog
        h,w = self.framesize
        for n in np.arange(self.nmax_frame, self.nmax_frame*0.1, self.nmax_frame/20.0):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, n-1)
            success, frame = self.cap.read()
            self.mogTV[fish].apply(frame[TVy1:TVy2,TVx1:TVx2,0], self.fgmaskTV[fish], self.learning_rate)
            self.mogSV[fish].apply(frame[SVy1:SVy2,SVx1:SVx2,0], self.fgmaskSV[fish], self.learning_rate)
        self.TVnoiseSize = self.noise_blob_sizeTV.GetValue()
        self.SVnoiseSize = self.noise_blob_sizeSV.GetValue()

    def OnSliderSpin(self, events, n=None):
        print 'OnSliderSpin'
        if n:
            self.sliderSpin.SetValue(n)
        else:
            n = self.sliderSpin.GetValue()
        self.slider.SetValue(n)
        self.curframe = n
        if self.namedWindow:
            self.updateFrame()

    def Onsavebtn(self, events):
        fish = self.fishname.GetValue()
        if not fish:
            return

        if fish in ['temp', 'mog']:
            dlg = wx.MessageDialog(self, "%s is reserved (temp, mog)" % fish, style=wx.OK)
            if dlg.ShowModal() == wx.ID_OK:
                dlg.Destroy()
                return

        if fish not in self.choiselist:
            self.choiselist.append(fish)
            self.RefreshChoise(self.targetfish, self.choiselist)
            self.ringpolyTVbtn.Disable()
            self.ringpolySVbtn.Disable()
            # self.savesetting(fish)
        else:
            if fish == 'all':
                registeredfish = [f for f in self.choiselist if f not in ('temp', 'mog', 'all')]
                print 'Saving for all fish (%s)' % registeredfish
            else:
                registeredfish = [fish]
            for fish in registeredfish:
                print 'Saving settins for %s' % fish
                self.savesetting(fish)
        self.updateCmap()
        self.Layout()
        self.Fit()

    def updateCmap(self):
        n = 4 if len(self.choiselist)<4 else len(self.choiselist)
        self.cmap4 = [map(int, (255*b,255*g,255*r)) for r,g,b,a
            # gist_rainbow rainbow hsv bwr
            in matplotlib.cm.gist_rainbow(np.linspace(0,255,n).astype(int))][::+1] 

    def Onremovebtn(self, events):
        fish = self.targetfish.GetStringSelection()
        if fish:
            dlg = wx.MessageDialog(self, 
                'Are you sure to remove this label (%s)?' % fish, style=wx.YES_NO)
            if dlg.ShowModal() == wx.ID_YES:
                print 'removing', fish, self.choiselist
                
                self.choiselist.remove(fish)
                if 'all' in self.choiselist and len(self.choiselist) <= 2: # less than 1 w/o 'all'
                    self.choiselist.remove('all')

                if self.fp:
                    settingfp = self.fp[:-3]+'pickle'
                    if os.path.exists(settingfp):
                        fname = os.path.basename(self.fp)
                        add_removePickledDict(settingfp, None, (fname, fish), remove=True)
                dlg.Destroy()

        self.RefreshChoise(self.targetfish, self.choiselist)
        if not self.choiselist:
            self.ringpolyTVbtn.Disable()
            self.ringpolySVbtn.Disable()

    def RefreshChoise(self, Choise, List):
        Choise.Clear()
        List.sort()
        for n, label in enumerate(List):
            Choise.Insert(label, n)

    def OnEventChoise(self, event):
        txt = self.event_label_choice.GetStringSelection()
        self.event_label.SetValue(txt)
        fish = self.targetfish.GetStringSelection()
        for k,timings in self.EventData[fish].items():
            if k == txt:
                print '%s: %s' % (k, ', '.join(
                    ['%02d:%02d (%d)' % (t/self.fps/60, t/self.fps%60, t) for t in timings]) )

    def OnEventRegister(self, event):
        fish = self.targetfish.GetStringSelection()
        if not fish:
            dlg = wx.MessageDialog(self, 
                "Select fish first\nbefore registering an event", style=wx.OK)
            if dlg.ShowModal() == wx.ID_OK:
                dlg.Destroy()
                return

        Label = self.event_label.GetValue()
        Selected = self.event_label_choice.GetStringSelection()

        # initialization as needed
        if Label not in self.event_label_list:
            self.event_label_list.append(Label)
            self.RefreshChoise(self.event_label_choice, self.event_label_list)
        if not self.EventData[fish].has_key(Selected) and Selected:
                self.EventData[fish][Selected] = []

        # add or remove an event
        if Selected and Selected == Label: # Event label needs to be selected at least

            if self.curframe in self.EventData[fish][Selected]: # remove existing one
                print 'Removing %s event at frame#%d' % (Selected, self.curframe)
                self.EventData[fish][Selected].remove(self.curframe)
                if len(self.EventData[fish][Selected]) == 0:
                    self.event_label_list.remove(Label)
                    self.RefreshChoise(self.event_label_choice, self.event_label_list)
                    self.EventData[fish].pop(Label)
            else: # add a new entery
                print 'Adding %s event at frame#%d' % (Selected, self.curframe)
                self.EventData[fish][Selected].append(self.curframe)

            print self.EventData[fish]

    def LoadExcel(self, fp):
        if self.fp: # if avi loaded already
            book = xlrd.open_workbook(fp)
            sh = book.sheet_by_index(0)

            self.EventData = {} # reset events, just in case
            temp = {} # to hold EventData temporally
            fishfound = []
            for ind in range(sh.nrows):
                fish, eventname = sh.row(ind)[0].value, sh.row(ind)[1].value
                eventtimes = [int(cell.value) for cell in sh.row(ind)[2:] if type(cell.value)==float]
                print fish, eventname, eventtimes
                if fish not in temp.keys():
                    temp[fish] = {}
                fishfound.append(fish)
                temp[fish][eventname] = eventtimes
            
            registeredfish = [f for f in self.choiselist if f not in ('temp', 'mog', 'all')]

            for fish in set(fishfound):
                if fish in registeredfish:
                    self.EventData[fish] = {}
                    print 'init', fish
                self.loadsetting(fish)
                self.savesetting(fish, Copying=temp[fish])

            self.OnChoice(None)

        else:
            print 'no avi opened yet.'

    def OnChoiceReplay_Track(self, event):
        if self.Replay_Online.GetStringSelection() == 'Track online':
            self.TrackOnline = True
            self.correctbtn.Disable()
        else:
            self.TrackOnline = False
            self.correctbtn.Enable()

    def OnChoice(self, events, settingfp=None):

        if not settingfp: # not pickle drop
            fish = self.targetfish.GetStringSelection()
            fname = os.path.basename(self.fp)
        else: # pickle drop
            fish = 'all'
            fname = os.path.basename(settingfp)[:-6] + 'avi'

        registeredfish = [f for f in self.choiselist if f not in ('temp', 'mog', 'all')]
        def init_fish(fish):
            if fish not in self.ringpolyTVDict.keys():
                self.ringpolyTVDict[fish] = {}
                self.ringpolyTVArray[fish] = None
            if fish not in self.ringpolySVDict.keys():
                self.ringpolySVDict[fish] = {}
                self.ringpolySVArray[fish] = None
            if fish not in self.InflowTubeTVdict.keys():
                self.InflowTubeTVdict[fish] = {}
                self.InflowTubeTVArray[fish] = None
            if fish not in self.InflowTubeSVdict.keys():
                self.InflowTubeSVdict[fish] = {}
                self.InflowTubeSVArray[fish] = None
            if fish not in self.TVtracking.keys():
                self.TVtracking[fish] = np.ones((self.nmax_frame,6)) * np.nan
                self.SVtracking[fish] = np.ones((self.nmax_frame,6)) * np.nan
            if fish not in self.EventData.keys():
                self.EventData[fish] = {}
            if fish not in self.ringpixel.keys():
                self.ringpixel[fish] = np.ones(self.nmax_frame) * np.nan
            if fish not in self.ROIcoords.keys():
                self.ROIcoords[fish] = []
            if fish not in self.mogTV.keys():
                self.OnMogParams(None, fish)
            self.playPausebtn.Enable()

        print 'self.getFishnamesfromPickle(fname)', self.getFishnamesfromPickle(fname)

        if fish == 'all':
            self.inflowTubebtn.Disable()
            self.ringpolyTVbtn.Disable()
            self.ringpolySVbtn.Disable()
            self.fishname.SetValue(fish)
            for f in registeredfish: # this ensure self.ROIcoords is fully updated.
                init_fish(f)
                if not settingfp:
                    self.loadsetting(f)
                    print 'load setting for ', f
                else: # for pickle_drop
                    self.loadsetting(f, settingfp, fname)
                    self.savesetting(f)
            self.updateFrame()
        elif fish in self.getFishnamesfromPickle(fname):
            init_fish(fish)
            self.loadsetting(fish)
            self.updateFrame()
            self.inflowTubebtn.Enable()
            self.ringpolyTVbtn.Enable()
            self.ringpolySVbtn.Enable()
            self.fishname.SetValue(fish)
        elif fish != 'temp' and fish:
            print 'init ', fish
            init_fish(fish)
            # self.updateFrame()
            self.inflowTubebtn.Enable()
            self.ringpolyTVbtn.Enable()
            self.ringpolySVbtn.Enable()
            self.fishname.SetValue(fish)

        # add 'all' to self.choiselist if more than 2 fish have been registered.
        if len(registeredfish)>=2:
            self.choiselist = list(set( (self.choiselist + ['all']) ))
            self.RefreshChoise(self.targetfish, self.choiselist) # this unselects the choice just made...
            if fish:
                self.targetfish.SetSelection(self.choiselist.index(fish)) # select back

    def OnresetTracking(self, events):
        fish = self.targetfish.GetStringSelection()
        dlg = wx.MessageDialog(self, 
            'Are you sure to reset tracking done for %s?' % fish, style=wx.YES_NO)
        if dlg.ShowModal() == wx.ID_YES:
            self.TVtracking[fish] = np.ones((self.nmax_frame,6)) * np.nan
            self.SVtracking[fish] = np.ones((self.nmax_frame,6)) * np.nan
            self.ringpixel[fish] = np.ones(self.nmax_frame) * np.nan

            self.ringpolyTVDict[fish] = {}
            self.ringpolyTVArray[fish] = None
            self.ringpolySVDict[fish] = {}
            self.ringpolySVArray[fish] = None

            self.InflowTubeTVdict[fish] = {}
            self.InflowTubeTVArray[fish] = None
            self.InflowTubeSVdict[fish] = {}
            self.InflowTubeSVArray[fish] = None 

            dlg.Destroy()

    def getSpinCtrls(self):
        TVx1 = self.ROITVx1.GetValue()
        TVy1 = self.ROITVy1.GetValue()
        TVx2 = self.ROITVx2.GetValue()
        TVy2 = self.ROITVy2.GetValue()
        TVH  = self.ROITVH .GetValue()

        SVx1 = self.ROISVx1.GetValue()
        SVy1 = self.ROISVy1.GetValue()
        SVx2 = self.ROISVx2.GetValue()
        SVy2 = self.ROISVy2.GetValue()
        SVx3 = self.ROISVx3.GetValue()
        SVy3 = self.ROISVy3.GetValue()

        return  TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3

    def loadsetting(self, fish, settingfp=None, fname=None):
        if not settingfp:
            pickled_dropped = False
            settingfp = self.fp[:-3]+'pickle'
            fname = os.path.basename(self.fp) # key for dict
        else:
            pickled_dropped = True

        with open(settingfp, 'rb') as f:
            params = pickle.load(f)

        if (fname, fish) in params.keys():
            # apply them
            thisfish = params[(fname, fish)]
            if 'TVx1' in thisfish.keys():
                TVx1,TVy1 = thisfish['TVx1'], thisfish['TVy1']
                TVx2,TVy2 = thisfish['TVx2'], thisfish['TVy2']
                TVH = thisfish['TVH']
                SVx1,SVy1 = thisfish['SVx1'], thisfish['SVy1']
                SVx2,SVy2 = thisfish['SVx2'], thisfish['SVy2']
                SVx3,SVy3 = thisfish['SVx3'], thisfish['SVy3']
                self.ROIcoords[fish] = TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3
                self.ROITVx1.SetValue(TVx1)
                self.ROITVy1.SetValue(TVy1)
                self.ROITVx2.SetValue(TVx2)
                self.ROITVy2.SetValue(TVy2)
                self.ROITVH.SetValue(TVH)
                self.ROISVx1.SetValue(SVx1)
                self.ROISVy1.SetValue(SVy1)
                self.ROISVx2.SetValue(SVx2)
                self.ROISVy2.SetValue(SVy2)
                self.ROISVx3.SetValue(SVx3)
                self.ROISVy3.SetValue(SVy3)

            if 'TVnoiseSize' in thisfish.keys():
                self.noise_blob_sizeTV.SetValue(thisfish['TVnoiseSize'])
                self.noise_blob_sizeSV.SetValue(thisfish['SVnoiseSize'])
            
            self.ringAppearochLevel.SetValue(thisfish['ringAppearochLevel'])

            mog_params = params[(fname, 'mog')]
            self.mog_history.SetValue(mog_params['history'])
            self.mog_nmixtures.SetValue(mog_params['nmixtures'])
            self.mog_backgroundRatio.SetValue(mog_params['backgroundRatio'])
            self.mog_noiseSigma.SetValue(mog_params['noiseSigma'])
            self.mog_learning_rate.SetValue(mog_params['learning_rate'])
            
            if 'depthCorrection' in mog_params.keys():
                self.depth.SetValue(mog_params['depthCorrection'])
            
            self.OnMogParams(None, fish) # init self.mog with the params in GUI

            self.preRange.SetValue(mog_params['preRange'])
            self.OnSpin(None)

            self.InflowTubeTVdict[fish] = thisfish['InflowTubeTVdict']
            self.InflowTubeSVdict[fish] = thisfish['InflowTubeSVdict']
            if thisfish['InflowTubeTVdict']:
                self.InflowTubeTVArray[fish] = interporatexyOverFrames(
                                    self.InflowTubeTVdict[fish], self.nmax_frame)
            if thisfish['InflowTubeSVdict']:
                self.InflowTubeSVArray[fish] = interporatexyOverFrames(
                                    self.InflowTubeSVdict[fish], self.nmax_frame)

            self.ringpolyTVDict[fish] = thisfish['ringpolyTVDict']
            self.ringpolySVDict[fish] = thisfish['ringpolySVDict']
            if thisfish['ringpolyTVDict']:
                self.ringpolyTVArray[fish] = interporatePolyOverFrames(
                                    self.ringpolyTVDict[fish], self.nmax_frame)
            if thisfish['ringpolySVDict']:
                self.ringpolySVArray[fish] = interporatePolyOverFrames(
                                    self.ringpolySVDict[fish], self.nmax_frame)

            self.EventData[fish] = thisfish['EventData']
            self.event_label_list = self.EventData[fish].keys()

            self.RefreshChoise(self.event_label_choice, self.event_label_list)

            trackingnpz = '%s_%s.npz' % (self.fp[:-4], fish)
            if os.path.exists(trackingnpz) and not pickled_dropped:
                data = np.load(trackingnpz)
                self.TVtracking[fish] = data['TVtracking']
                self.SVtracking[fish] = data['SVtracking']
                self.ringpixel[fish] = data['ringpixel']

    def savesetting(self, fish, Copying=False):
        # save the setting of currently selected fish
        selected = self.targetfish.GetStringSelection()
        if selected == 'all':
            TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = self.ROIcoords[fish]
        else:
            TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = self.getSpinCtrls()

        ringpolyTVDict = {} if fish not in self.ringpolyTVDict.keys() else self.ringpolyTVDict[fish]
        ringpolySVDict = {} if fish not in self.ringpolySVDict.keys() else self.ringpolySVDict[fish]
        InflowTubeTVdict = {} if fish not in self.InflowTubeTVdict.keys() else self.InflowTubeTVdict[fish]
        InflowTubeSVdict = {} if fish not in self.InflowTubeSVdict.keys() else self.InflowTubeSVdict[fish]
        
        if Copying:
            EventData = Copying
        else:
            EventData = {} if fish not in self.EventData.keys() else self.EventData[fish]
        
        # avi and fish combination
        roi_params = {
            'TVx1' : TVx1, 'TVy1' : TVy1, 'TVx2' : TVx2, 'TVy2' : TVy2, 'TVH'  : TVH,
            'SVx1' : SVx1, 'SVy1' : SVy1, 'SVx2' : SVx2, 'SVy2' : SVy2, 'SVx3' : SVx3, 'SVy3' : SVy3,
            'ringpolyTVDict' : ringpolyTVDict, 'ringpolySVDict' : ringpolySVDict,
            'InflowTubeTVdict' : InflowTubeTVdict, 'InflowTubeSVdict' : InflowTubeSVdict,
            'TVnoiseSize' : self.TVnoiseSize, 'SVnoiseSize' : self.SVnoiseSize, 
            'EventData' : EventData,  # in case different fish get different odors CS+/CS-
            'ringAppearochLevel' : self.ringAppearochLevel.GetValue(),  # in case fish are so different in size 
            'depthCorrection' : self.depth.GetValue(),  # this changes with the camera position and angle
            }
        # common for avi file
        history, nmixtures, backgroundRatio, noiseSigma, learning_rate = self.getMogParams()
        mog_params = {
            'history' : history,
            'nmixtures' : nmixtures,
            'backgroundRatio' : backgroundRatio,
            'noiseSigma' : noiseSigma,
            'learning_rate' : learning_rate,
            'preRange' : self.preRange.GetValue(),
            'fps' : self.fps,
            }
        fname = os.path.basename(self.fp) # key for dict
        settingfp = self.fp[:-3]+'pickle'

        if not os.path.exists(settingfp):
            with open(settingfp, 'wb') as f:
                settings = {(fname, fish): roi_params, (fname, 'mog'): mog_params}
                pickle.dump(settings, f)
        else:
            add_removePickledDict(settingfp, roi_params, (fname, fish))
            add_removePickledDict(settingfp, mog_params, (fname, 'mog'))

        trackingnpz = '%s_%s.npz' % (self.fp[:-4], fish)
        np.savez_compressed(trackingnpz, 
                ringpixel=self.ringpixel[fish],
                TVtracking=self.TVtracking[fish], 
                SVtracking=self.SVtracking[fish],
                InflowTubeTVArray=self.InflowTubeTVArray[fish], 
                InflowTubeSVArray=self.InflowTubeSVArray[fish], 
                ringpolyTVArray=self.ringpolyTVArray[fish],
                ringpolySVArray=self.ringpolySVArray[fish],
                TVbg=self.bg[TVy1:TVy2, TVx1:TVx2],
                SVbg=self.bg[SVy1:SVy2, SVx1:SVx2],
                )

        if self.needMatfile.IsChecked():
            trackingmat = '%s_%s.mat' % (self.fp[:-4], fish)
            datadict = {
                    'ringpixel':self.ringpixel[fish],
                    'TVtracking':self.TVtracking[fish], 
                    'SVtracking':self.SVtracking[fish],
                    'InflowTubeTVArray':self.InflowTubeTVArray[fish], 
                    'InflowTubeSVArray':self.InflowTubeSVArray[fish], 
                    'ringpolyTVArray':self.ringpolyTVArray[fish],
                    'ringpolySVArray':self.ringpolySVArray[fish],
                    'TVbg' : self.bg[TVy1:TVy2, TVx1:TVx2],
                    'SVbg' : self.bg[SVy1:SVy2, SVx1:SVx2],
                    }
            sio.savemat(trackingmat, datadict, oned_as='row', do_compression=True)

    #       wx self;  |<=   cv2 callback   =>|
    def onMouse(self,  event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONUP:
            fish = self.targetfish.GetStringSelection()
            TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = self.getSpinCtrls()
            fish = self.targetfish.GetStringSelection()
            n = self.curframe #self.slider.GetValue()
            print 'x,y=(%d,%d)' % (x,y)
            if self.ringpolyTV or self.ringpolySV:
                self.clicks.append((x,y))
                if len(self.clicks) > 8: # replace the nearest point
                    temp = np.array(self.clicks)
                    d = np.abs(temp[:-1,:] - temp[-1,:]).sum(axis=1)
                    ind = np.abs(temp[:-1,:] - temp[-1,:]).sum(axis=1).argmin()
                    if d.min() < 40:
                        self.clicks.pop(ind)
                    else:
                        self.clicks.pop(0)

            elif self.inflowTubebtn.GetValue():
                if TVx1<=x<=TVx2 and TVy1<=y<=TVy2:
                    self.InflowTubeTVdict[fish][n] = (x,y)
                    self.InflowTubeTVArray[fish] = interporatexyOverFrames(
                                    self.InflowTubeTVdict[fish], self.nmax_frame)
                elif SVx1<=x<=SVx2 and SVy1<=y<=SVy2:
                    self.InflowTubeSVdict[fish][n] = (x,y)
                    self.InflowTubeSVArray[fish] = interporatexyOverFrames(
                                    self.InflowTubeSVdict[fish], self.nmax_frame)
                else:
                    print 'outside of any ROIs'

            elif self.correctbtn.GetValue():
                # print flags
                if flags == cv2.EVENT_FLAG_CTRLKEY or flags==9: # cv2.EVENT_FLAG_CTRLKEY is 8L somehow but I get 9...
                    if TVx1<=x<=TVx2 and TVy1<=y<=TVy2:
                        print 'head x,y corrected'
                        self.TVtracking[fish][n,3:5] = (x,y)
                        self.OnSliderSpin(None, self.curframe) # refresh window
                    else:
                        print 'outside of any ROIs'
                else:
                    if TVx1<=x<=TVx2 and TVy1<=y<=TVy2:
                        self.TVtracking[fish][n,:2] = (x,y)
                        self.curframe += self.frameStepSpin.GetValue()
                    elif SVx1<=x<=SVx2 and SVy1<=y<=SVy2:
                        self.SVtracking[fish][n,:2] = (x,y)
                        self.curframe += self.frameStepSpin.GetValue()
                    else:
                        print 'outside of any ROIs'
                    self.OnSliderSpin(None, self.curframe)
            self.updateFrame()

    def OnRingpolybtn(self, events):
        fish = self.targetfish.GetStringSelection()
        n = self.slider.GetValue()
        if fish:
            btnId = events.GetId()
            if btnId == self.ID_ringpolyTV:
                if self.ringpolyTV and len(self.clicks) == 8:
                    self.ringpolyTV = False
                    self.ringpolyTVbtn.SetBackgroundColour(wx.NullColour)
                    print 'ringpolyTV %s, %d' % (self.clicks, n)
                    self.ringpolyTVDict[fish][n] = self.clicks
                    self.ringpolyTVArray[fish] = interporatePolyOverFrames(
                                    self.ringpolyTVDict[fish], self.nmax_frame)
                else:
                    self.ringpolyTV = True
                    self.ringpolyTVbtn.SetBackgroundColour('yellow')
                    self.clicks = []

            elif btnId == self.ID_ringpolySV:
                if self.ringpolySV:
                    self.ringpolySV = False
                    self.ringpolySVbtn.SetBackgroundColour(wx.NullColour)
                    print 'ringpolySV %s, %d' % (self.clicks, n)
                    self.ringpolySVDict[fish][n] = self.clicks
                    self.ringpolySVArray[fish] = interporatePolyOverFrames(
                                    self.ringpolySVDict[fish], self.nmax_frame)
                else:
                    self.ringpolySV = True
                    self.ringpolySVbtn.SetBackgroundColour('yellow')
                    self.clicks = []

    def LoadAvi(self, events, fp):
        self.namedWindow = True
        self.OnnamedWinbtn(None)

        self.fp = fp
        basefolder = os.path.dirname(fp)
        fname = os.path.basename(fp)

        self.filehistory.AddFileToHistory(fp)
        self.filehistory.Save(self.config)
        self.config.Flush()

        self.cap = cv2.VideoCapture(fp)
        success, frame = self.cap.read()
        if success:
            self.topBorder = int(frame.shape[0]/2)
            GL = self.groundLevel.GetValue()
            if GL>0: # clip the bottom
                frame = frame[:GL,:,:]
            self.nmax_frame = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            self.fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
            self.framesize = frame.shape[:2]
            self.clipfp = False
            self.lastContour = None

            self.ringpolyTVArray = {} # key is fish name (top, btm, etc)
            self.ringpolySVArray = {}
            self.ringpolyTVDict = {}
            self.ringpolySVDict = {}
            self.InflowTubeTVdict = {}
            self.InflowTubeSVdict = {}
            self.InflowTubeTVArray = {}
            self.InflowTubeSVArray = {}
            self.TVtracking = {}
            self.SVtracking = {}
            self.ringpixel = {}
            self.EventData = {}
            self.ROIcoords = {}
            self.fgmaskTV = {}
            self.fgmaskSV = {}
            self.mogTV = {}
            self.mogSV = {}

            self.namedWinbtn.Enable()
            # self.playPausebtn.Enable()
            self.resetTracking.Enable()
            self.SetTitle( "%s" % fname )
            self.slider.Enable()
            self.sliderSpin.Enable()
            self.slider.SetMax(self.nmax_frame-1)
            self.sliderSpin.SetRange(0,self.nmax_frame-1)
            print '%s loaded (%d frames).' % (fname, self.nmax_frame)

            # stationary background
            bg = np.zeros(self.framesize)
            for n in np.linspace(self.nmax_frame-1, 0, 20):
                self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, n)
                _, frame = self.cap.read()
                if GL>0:
                    bg = np.dstack((bg,frame[:GL,:,0]))
            self.bg = np.median(bg, axis=2).astype(np.int)

        else:
            dlg = wx.MessageDialog(self, 'File did not load!', 'IO Error',
                                       wx.OK | wx.ICON_INFORMATION )
            dlg.ShowModal()
            dlg.Destroy()
            return

        # loading previous fish names
        self.choiselist = self.getFishnamesfromPickle(fname)
        
        self.OnMogParams(None, 'temp') # init self.mog with the params in GUI

        self.RefreshChoise(self.targetfish, self.choiselist)
        self.OnChoice(None)

        self.updateCmap()
        self.OnSpin(None)

        self.Layout()
        self.Fit()

    def Onfilterheadxy(self, events):
        n = self.curframe
        fish = self.targetfish.GetStringSelection()
        preRange = self.preRange.GetValue() 
        rng = np.arange(n-preRange, n+preRange)
        headx = self.TVtracking[fish][rng,3]
        heady = self.TVtracking[fish][rng,4]
        
        X,Y = filterheadxy(headx,heady)

        self.TVtracking[fish][rng,3] = X
        self.TVtracking[fish][rng,4] = Y

    def getFishnamesfromPickle(self, fname, settingfp=None):
        if not settingfp:
            settingfp = self.fp[:-3]+'pickle'
        else:
            fname = os.path.basename(settingfp)[:-6] + 'avi'

        if os.path.exists(settingfp):
            with open(settingfp, 'rb') as f:
                data = pickle.load(f)
            return [fs for fl,fs in data.keys() if fl == fname and fs != 'mog']
        else:
            return []

    def updateFrame(self):
        n = self.curframe
        self.sb.SetStatusText('Time %02d:%02d' % (n/self.fps/60, n/self.fps%60))
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, n-1)
        success, frame = self.cap.read()
        if success:
            h,w = self.framesize
            frame = frame[:h,:,:]
            trackingmode = self.trackingmode.GetCurrentSelection()

            fish = self.targetfish.GetStringSelection()

            if fish == 'all':
                drawn = frame.copy()
                for eachfish in [f for f in self.choiselist if f not in ('all', 'mog', 'temp')]:

                    ROIcoords = self.ROIcoords[eachfish]
                    TVContour,SVContour,newX,newY,newY2,tvx,tvy,svx,svy,x0,y0,headx,heady = self.track(
                                frame,eachfish,n,w,h,trackingmode,ROIcoords)
                    if self.namedWindow:
                        drawn = self.draw(eachfish,drawn,n,TVContour,SVContour,newX,newY,newY2,w,h,
                                        tvx,tvy,svx,svy,ROIcoords,x0,y0,headx,heady)
            else:
                ROIcoords = self.getSpinCtrls()
                if not fish:
                    fish = 'temp'
                TVContour,SVContour,newX,newY,newY2,tvx,tvy,svx,svy,x0,y0,headx,heady = self.track(
                                        frame,fish,n,w,h,trackingmode,ROIcoords)
                if self.namedWindow:
                    drawn = self.draw(fish,frame.copy(),n,TVContour,SVContour,newX,newY,newY2,w,h,
                                tvx,tvy,svx,svy,ROIcoords,x0,y0,headx,heady)
            
            if self.namedWindow:
                cv2.imshow('Tracking', drawn)
            if self.clipfp:
                self.writer.write(drawn)
        else:
            self.Onplaybtn(None)


    def track(self, frame, fish, n, w, h, trackingmode, ROIcoords):
        
        TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = ROIcoords

        def findFish(fg):
            contours, hierarchy = cv2.findContours( fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
            contours = [c for c in contours if cv2.contourArea(c.astype(int)) > self.TVnoiseSize]
            cntrs = [np.dstack((c[:,0,0] + TVx1, c[:,0,1] + TVy1)).reshape(c.shape) for c in contours]
            centers = [f.mean(axis=0)[0].tolist() for f in cntrs]
            areas = [cv2.contourArea(c.astype(int)) for c in cntrs]
            return contours, cntrs, centers, areas

        def pickbydistandsize(centers, areas):
            bydist = [np.sqrt( ((prepos-c)**2).sum() ) for c in centers] # smaller better
            bysize = [abs(a-avgsize) for a in areas]
            weighted = [s+d**1.5 for s,d in zip(bysize, bydist)]
            bysizedist = np.argmin( weighted )
            # if bysizedist:
                # print n, bydist, bysize, weighted, bysizedist
            return bysizedist
        
        # //TOP view
        self.mogTV[fish].apply(frame[TVy1:TVy2,TVx1:TVx2,0], self.fgmaskTV[fish], self.learning_rate)
        
        k = int(self.ringAppearochLevel.GetValue() / 4) # 12-20 works for me
        if self.ringpolyTVArray[fish] is not None:
            rx, ry, rw, rh, rang = self.ringpolyTVArray[fish][n,:].astype(int)
            cv2.ellipse(self.fgmaskTV[fish], ((rx-TVx1,ry-TVy1), (rw,rh), rang), 0, k)

        TVContour, TVpos, TVarea = None, None, None
        x0, y0, headx, heady = None, None, None, None
        if self.TrackOnline:
            centers = None

            if trackingmode != 2:   # 0 use both; 1 force mog; 2 force bg
                contours, cntrs, centers, areas = findFish(self.fgmaskTV[fish].copy())

            if (not centers and trackingmode != 1) or trackingmode == 2:
                _temp = cv2.subtract(self.bg[TVy1:TVy2,TVx1:TVx2], frame[TVy1:TVy2,TVx1:TVx2,0].astype(np.int))
                _temp[_temp<12] = 0
                _temp = _temp.astype(np.uint8)
                if self.showfgmask:
                    self.fgmaskTV[fish] = _temp.copy()
                contours, cntrs, centers, areas = findFish(_temp.copy())
            
            # split fish under ring? repair with erode
            st = n-3 if n-3 > 0 else 0
            if np.isnan(self.TVtracking[fish][st:n, :2]).all():
                prepos = np.array([TVy1+(TVy2-TVy1)/2, TVx1+(TVx2-TVx1)/2]) # center of ROI
            else:
                prepos = np.ma.masked_invalid(self.TVtracking[fish][st:n, :2]).mean(axis=0)
            st = n-50 if n-50 > 0 else 0
            avgsize = np.mean(np.ma.masked_invalid( self.TVtracking[fish][st:, 2] ))

            if areas and self.connectSplitBlobs.IsChecked():

                bysizedist = pickbydistandsize(centers, areas)

                if avgsize*0.8 < areas[bysizedist] < avgsize*1.2:
                    # print 'simiar size (+/-20%), pos fish found ', areas[bysizedist]
                    pass

                elif self.ringpolyTVArray[fish] is not None:

                    # merge from the closest one
                    bx1,by1 = centers[bysizedist]
                    d2blob = [np.sqrt((bx-bx1)**2 + (by-by1)**2) for bx,by in centers]

                    merged = contours[bysizedist].copy()
                    good = [merged]
                    tobemerged = None
                    for ind in np.argsort(d2blob)[1:]:
                        if d2blob[ind] < min((TVx2-TVx1), (SVx2-SVx1))/2:
                            tobemerged = np.concatenate([merged, contours[ind]]) # for computing total area (not convexhull)
                            mergedsize = cv2.contourArea(tobemerged.astype(int))
                            if mergedsize < avgsize*1.2:
                                merged = tobemerged
                                good.append(contours[ind])

                    if tobemerged is not None:
                        fishfg = np.zeros_like(frame[TVy1:TVy2, TVx1:TVx2,0])
                        for g in good:
                            cv2.fillConvexPoly(fishfg, g, 1)
                        vx, vy, x0, y0 = cv2.fitLine(tobemerged,cv2.cv.CV_DIST_HUBER, 0, 0.01, 0.01)
                        cx = tobemerged[:,0,0]
                        cy = tobemerged[:,0,1]
                        c = y0 - (vy/vx) * x0
                        x1, x2 = cx.min(), cx.max()
                        # print 'vy/vx', vy/vx
                        if x2-x1 > 15:
                            y1 = x1 * (vy/vx) + c
                            y2 = x2 * (vy/vx) + c
                            # print 'horizontal',x1,int(y1[0]),x2,int(y2[0])
                        else: # vertical fish
                            y1, y2 = cy.min(), cy.max()
                            x1 =  (y1-c) / (vy/vx) 
                            x2 =  (y2-c) / (vy/vx) 
                            # print 'vertical',int(x1[0]),y1,int(x2[0]),y2

                        ringfg = np.zeros_like(frame[TVy1:TVy2, TVx1:TVx2,0])
                        cv2.ellipse(ringfg, ((rx-TVx1,ry-TVy1), (rw,rh), rang), 1, k)

                        bridge = np.zeros_like(frame[TVy1:TVy2, TVx1:TVx2,0])
                        cv2.line(bridge, (x1,y1), (x2,y2), 1, int(k/2))
                        bridged = (ringfg & bridge) + fishfg

                        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, np.ones((k/2,k/2)))

                        if self.showfgmask:
                            self.fgmaskTV[fish] = 255*(bridged.copy())
                        
                        # print 'b4 repair size', areas[bysizedist],
                        contours, cntrs, centers, areas = findFish(bridged.copy())
                        if areas:
                            bysizedist = pickbydistandsize(centers, areas)
                            # print '-> repaired size', areas[bysizedist]


            # now picking the fish contour
            self.TVtracking[fish][n,:2] = (np.nan, np.nan)
            self.TVtracking[fish][n,2] = np.nan
            tvx,tvy = None, None
            if centers:
                
                bydist = [np.sqrt( ((prepos-c)**2).sum() ) for c in centers]
                bysize = [a-avgsize if (a-avgsize)>0 else (avgsize-a)/3 for a in areas]
                ind = np.argmin( [s+5*d for s,d in zip(bysize, bydist)] )
                
                TVContour, TVpos, TVarea = cntrs[ind], centers[ind], areas[ind]
                self.TVtracking[fish][n,:2] = TVpos
                self.TVtracking[fish][n,2] = TVarea
                tvx,tvy = map(int, TVpos)
                # print 'TVarea avgsize areas', TVarea, int(avgsize)
                
                # orientation esstimation here
                if contours[ind].shape[0] > 5 and avgsize*0.15<TVarea<avgsize*1.5:

                    vx, vy, x0, y0 = cv2.fitLine(contours[ind], cv2.cv.CV_DIST_HUBER, 0, 0.01, 0.01)
                    cx, cy = contours[ind][:,0,0], contours[ind][:,0,1]
                    c = y0 - (vy/vx) * x0
                    x1, x2 = cx.min(), cx.max()
                    if x2-x1 > 15:
                        y1 = x1 * (vy/vx) + c
                        y2 = x2 * (vy/vx) + c
                        step = 1 if (x2-x1) < 20 else (x2-x1)/20
                        X = np.arange(x1,x2,step)
                        Y = X*(vy/vx) + c
                    else: # vertical fish
                        y1, y2 = cy.min(), cy.max()
                        x1 =  (y1-c) / (vy/vx) 
                        x2 =  (y2-c) / (vy/vx) 
                        step = 1 if (y2-y1) < 20 else (y2-y1)/20
                        Y = np.arange(y1,y2,step)
                        X = (Y-c) / (vy/vx)

                    # guess head location by body thickness
                    overlap = []
                    for _x, _y in zip(X,Y):
                        canvas = np.zeros_like(self.fgmaskTV[fish])
                        cv2.circle(canvas, (int(_x),int(_y)), 12, 255, -1)  # 9 is hardcoded radius 
                        overlap.append(self.fgmaskTV[fish][canvas>0].sum())
                        # print _x, _y, overlap
                    
                    ov = np.array(overlap)
                    # simple peak usually works the best
                    ind_max = ov.argmax()
                    # delta overlap typically has one broad peak around head
                    dov = np.diff(ov)
                    ind_diff = dov.argmax()
                    # slope based
                    ind_half = int(ov.size/2)
                    if ov[:ind_half].sum() > ov[ind_half:].sum():
                        ind_slope = int(ov.size/4)
                    else:
                        ind_slope = int(ov.size/4*3)
                    # democratic decision
                    # print [ind_max, ind_diff, ind_slope]
                    head_ind = np.median([ind_max, ind_diff, ind_slope])

                    headx = X[head_ind] + TVx1
                    heady = Y[head_ind] + TVy1
                    x0 += TVx1
                    y0 += TVy1

                    self.TVtracking[fish][n,3] = headx 
                    self.TVtracking[fish][n,4] = heady 

        else: # replay mode
            tvx,tvy = self.TVtracking[fish][n,:2].astype(np.int)
            headx,heady = self.TVtracking[fish][n,3:5].astype(np.int)
            x0, y0 = tvx,tvy


        #//bottom view
        self.mogSV[fish].apply(frame[SVy1:SVy2,SVx1:SVx2,0], self.fgmaskSV[fish], self.learning_rate)

        # now adjust the ROI for SV using the recent TVpos tvx,tvy
        tB = self.topBorder
        if tvx and TVx2-TVx1 >= TVy2-TVy1: # TopView horizontal case
            factor = self.depth.GetValue()
            if TWOCHOICE:
                newY =  int(SVy2 -        (SVy2-SVy3) * (tvx-TVx1) / (TVx2-TVx1) )
                newY2 = int(SVy1 + factor*(SVy2-SVy3) * (tvx-TVx1) / (TVx2-TVx1) )
            else:
                newY =  int(SVy3        + (SVy2-SVy3) * (tvx-TVx1) / (TVx2-TVx1) )
                newY2 = int(SVy1 + factor*(SVy2-SVy3) - factor*(SVy2-SVy3) * (tvx-TVx1) / (TVx2-TVx1) )
            self.fgmaskSV[fish][:newY2-SVy1, :] = 0
            self.fgmaskSV[fish][newY-SVy1:, :] = 0

            if SVx3 > w/2: # right side of screen
                if TWOCHOICE:
                    newX = int(SVx1 + (SVx2 - SVx3) * (tvx-TVx1) / (TVx2-TVx1) )
                    self.fgmaskSV[fish][:, :newX-SVx1] = 0
                else:
                    newX = int(SVx3 + (SVx2 - SVx3) * (tvx-TVx1) / (TVx2-TVx1) )
                    self.fgmaskSV[fish][:, newX-SVx1:] = 0
            
            else:
                if TWOCHOICE:
                    newX = int(SVx1 + (SVx3 - SVx1) * (tvx-TVx1) / (TVx2-TVx1) )
                    self.fgmaskSV[fish][:, :newX-SVx1] = 0
                else:
                    newX = int(SVx3 - (SVx3 - SVx1) * (tvx-TVx1) / (TVx2-TVx1) )
                    self.fgmaskSV[fish][:, :newX-SVx1] = 0
            


        elif tvy and TVx2-TVx1 < TVy2-TVy1:  # TopView vertical case
            newY = int(SVy3 + (SVy2-SVy3) * (tvy - TVy1) / (TVy2-TVy1) )
            self.fgmaskSV[fish][newY-SVy1:, :] = 0
            if SVx3 > w/2: # right side of screen
                newX = int(SVx3 + (SVx2 - SVx3) * (tvy - TVy1) / (TVy2-TVy1) )
                self.fgmaskSV[fish][:, newX-SVx1:] = 0
            else:
                newX = int(SVx3 - (SVx3 - SVx1) * (tvy - TVy1) / (TVy2-TVy1) )
                self.fgmaskSV[fish][:, :newX-SVx1] = 0
        
        else:
            newX, newY, newY2 = None, None, None

        if self.ringpolySVArray[fish] is not None:
            rx,ry,rh,rw,rangle = self.ringpolySVArray[fish][n,:]
            # get the ringpixel b4 removing fg inside ring
            if self.TrackOnline:
                # in case the ring moves aound, this needs to be dynamic...
                ringmask = np.zeros(self.fgmaskSV[fish].shape, dtype=np.uint8)
                cv2.ellipse(ringmask, ((rx-SVx1,ry-SVy1), (rh,rw), rangle), 1, -1)
                ringmask = ringmask>0
                self.ringpixel[fish][n] = self.fgmaskSV[fish][ringmask].sum() / 255
            # now remove ring pixels
            ringBtm = int(ry+rh/2-SVy1)
            self.fgmaskSV[fish][:ringBtm, :] = 0
        else:
            ringBtm = None
        

        SVContour, SVpos, SVarea = None, None, None
        if self.TrackOnline:
            centers = None
            if trackingmode != 2:   # 0 use both; 1 force mog; 2 force bg
                contours, hierarchy = cv2.findContours( self.fgmaskSV[fish].copy(),
                                            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
                cntrs = [np.dstack((c[:,0,0] + SVx1, c[:,0,1] + SVy1)).reshape(c.shape)
                            for c in contours if cv2.contourArea(c.astype(int)) > self.SVnoiseSize]
                centers = [f.mean(axis=0)[0] for f in cntrs]
                areas = [cv2.contourArea(c.astype(int)) for c in cntrs]
                if not centers and self.namedWindow:
                    __areas = [cv2.contourArea(c.astype(int)) for c in contours]
                    if __areas:
                        print n, 'SV', [a for a in __areas if a]

            if (not centers and trackingmode != 1) or trackingmode == 2:
                _temp = cv2.absdiff(
                    frame[SVy1:SVy2,SVx1:SVx2,0].astype(np.int), self.bg[SVy1:SVy2,SVx1:SVx2])
                _temp[_temp<20] = 0
                if ringBtm:
                    _temp[:ringBtm, :] = 0
                if tvx:
                    _temp[newY-SVy1:, :] = 0
                    if SVx3 > w/2:
                        _temp[:, newX-SVx1:] = 0
                    else:
                        _temp[:, :newX-SVx1] = 0
                _temp = _temp.astype(np.uint8)
                if self.showfgmask:
                    self.fgmaskSV[fish] = _temp.copy()
                contours, hierarchy = cv2.findContours(_temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cntrs = [np.dstack((c[:,0,0] + SVx1, c[:,0,1] + SVy1)).reshape(c.shape)
                            for c in contours if cv2.contourArea(c.astype(int)) > self.TVnoiseSize]
                centers = [f.mean(axis=0)[0] for f in cntrs]
                areas = [cv2.contourArea(c.astype(int)) for c in cntrs]

            self.SVtracking[fish][n,:2] = (np.nan, np.nan)
            self.SVtracking[fish][n,2] = np.nan
            svx,svy = None, None
            if centers:
                if np.isnan(self.SVtracking[fish][st:n, :2]).all():
                    prepos = np.array([3*h/2, w/2])
                    lostlast = True
                else:
                    lostlast = False
                    prepos = np.ma.masked_invalid(self.SVtracking[fish][st:n, :2]).mean(axis=0)
                st = n-50 if n-50 > 0 else 0
                _SVavgsize = np.mean(np.ma.masked_invalid( self.SVtracking[fish][st:n, 2] ))
                bydist = [np.sqrt( ((prepos-c)**2).sum() ) for c in centers]
                bysize = [a-_SVavgsize if (a-_SVavgsize)>0 else (_SVavgsize-a)/3 for a in areas]

                # closest by size and dist
                okdist = [_n for _n,d in enumerate(bydist) if d<100]
                ind = np.argmin( [s+5*d for s,d in zip(bysize, bydist)] )
                if ind in okdist or n<100 or lostlast:
                    SVContour, SVpos, SVarea = cntrs[ind], centers[ind], areas[ind]
                    self.SVtracking[fish][n,:2] = SVpos
                    self.SVtracking[fish][n,2] = SVarea
                    svx,svy = map(int, SVpos)
        
        else:
            svx,svy = self.SVtracking[fish][n,:2].astype(np.int)
        # # bottom//

        return TVContour,SVContour,newX,newY,newY2,tvx,tvy,svx,svy,x0,y0,headx,heady


    def draw(self,fish,frame,n,TVContour,SVContour,newX,newY,newY2,w,h,tvx,tvy,svx,svy,ROIcoords,x0,y0,headx,heady):

        if fish != 'temp':
            color = self.cmap4[self.choiselist.index(fish)]
        else:
            color = self.cmap4[0]
        color2 = [255-c for c in color]

        if self.showfgmask:
            cv2.imshow('fgmaskTV', self.fgmaskTV[fish])
            cv2.imshow('fgmaskSV', self.fgmaskSV[fish])
        
        # ROI frames etc
        TVx1,TVy1,TVx2,TVy2,TVH,SVx1,SVy1,SVx2,SVy2,SVx3,SVy3 = ROIcoords
        cx,cy = (TVx1+TVx2)/2, (TVy1+TVy2)/2
        boxTV = cv2.cv.BoxPoints(((cx,cy), (TVx2-TVx1, TVy2-TVy1), TVH))
        cv2.drawContours(frame, [np.int0(boxTV)], 0, color, thickness=2)
        cv2.rectangle(frame, (SVx1, SVy1), (SVx2, SVy2), color, thickness=2)
        cv2.line(frame, (SVx3, SVy3+20), (SVx3, SVy3-20), color) # SVx3
        cv2.line(frame, (SVx3-20, SVy3), (SVx3+20, SVy3), color) # SVx3

        # overlay of edge detection for fish outline
        fgmaskTV = cv2.Canny(self.fgmaskTV[fish].copy(), 1, 254) # fgmask is either 0 or 255
        frame[TVy1:TVy2,TVx1:TVx2,:][fgmaskTV>0,:] = color2

        # drawing trajectory
        if fish:
            TVx = interp_nan(self.TVtracking[fish][:,0].copy())
            TVy = interp_nan(self.TVtracking[fish][:,1].copy())
            SVx = interp_nan(self.SVtracking[fish][:,0].copy())
            SVy = interp_nan(self.SVtracking[fish][:,1].copy())
            rp = interp_nan(self.ringpixel[fish].copy())

            st = n-150 if n > 150 else 0
            if TVx is not None and TVy is not None:
                cv2.polylines(frame, [np.array([TVx[st:n],TVy[st:n]]).astype(np.int32).T], False, color)
            if SVx is not None and SVy is not None:
                cv2.polylines(frame, [np.array([SVx[st:n],SVy[st:n]]).astype(np.int32).T], False, color)
        else:
            TVx, TVy = None, None

        # swim direction
        if self.ringpolySVArray[fish] is not None:
            rz = self.ringpolySVArray[fish][n,1].astype(np.int)
            dt = 3
            a = TVx[n]-TVx[n-dt]
            b = TVy[n]-TVy[n-dt]
            c = SVy[n]-SVy[n-dt]
            # x-x0   y-y0   z-z0
            # ---- = ---- = ----
            #   a      b      c
            # solve them for z = rz. x0,y0,z0 are tvx, tvy, svy
            # x = (a * (rz-svy)) / c + tvx
            if c<-2/30.0*self.fps: # cross between water surface and swim direction line when going upward
                water_x = (a * (rz-SVy[n]) / c) + TVx[n]
                water_y = (b * (rz-SVy[n]) / c) + TVy[n]

                if -np.isinf(water_x) and -np.isinf(water_y) and -np.isnan(water_x) and -np.isnan(water_y):
                    water_x = int(water_x)
                    water_y = int(water_y)

                    if TVx1<water_x<TVx2 and TVy1<water_y<TVx2 and -np.isinf(water_x) and -np.isinf(water_y):
                        cv2.line(frame, (int(TVx[n]), int(TVy[n])), (water_x, water_y), color2, thickness=2)
                        # print '%2.2f %2.2f %d %d %d %d' % (a, c, (rz-SVy[n]), (rz-SVy[n]) / c, water_x, water_y)


        # draw the detected Contours
        if self.TrackOnline:
            cv2.drawContours(frame, TVContour, -1, color)
            cv2.drawContours(frame, SVContour, -1, color)
        # cross hairs for position
        if tvx:
            cv2.line(frame, (tvx-9,tvy), (tvx+9,tvy), color, thickness=2)
            cv2.line(frame, (tvx,tvy-9), (tvx,tvy+9), color, thickness=2)
            # update clip border when tvx is available
            cv2.line(frame, (newX,SVy1), (newX,SVy2), color, thickness=1)
            cv2.line(frame, (SVx1,newY), (SVx2,newY), color, thickness=1)
            cv2.line(frame, (SVx1,newY2), (SVx2,newY2), color, thickness=1)
        if svx:
            cv2.line(frame, (svx-9,svy), (svx+9,svy), color, thickness=2)
            cv2.line(frame, (svx,svy-9), (svx,svy+9), color, thickness=2)

        if self.InflowTubeTVArray[fish] is not None:
            xinf,yinf = self.InflowTubeTVArray[fish][n]
            cv2.line(frame, (xinf-9,yinf), (xinf+9,yinf), color)
            cv2.line(frame, (xinf,yinf-9), (xinf,yinf+9), color)
        if self.InflowTubeSVArray[fish] is not None:
            xinf,yinf = self.InflowTubeSVArray[fish][n]
            cv2.line(frame, (xinf-9,yinf), (xinf+9,yinf), color)
            cv2.line(frame, (xinf,yinf-9), (xinf,yinf+9), color)

        if headx:  # TVx1,TVy1,TVx2,TVy2
            cv2.circle(frame, (int(x0),int(y0)), 3, color2, 2)
            cv2.circle(frame, (int(headx),int(heady)), 12, color2, 2)
        
        tB = self.topBorder
        # plotting
        if TVx is not None and TVy is not None:
            t = np.linspace(0, w, self.nmax_frame)
            # event timings
            curpos = int(w*n/self.nmax_frame)
            cv2.line(frame, (curpos,tB+0), (curpos,tB+60), color2)
            if fish is not 'temp':
                _keys = self.EventData[fish].keys()
                CSs = [k for k in _keys if k.startswith('CS')]
                USs = [k for k in _keys if k.startswith('US')]
                others = set(_keys) - set(CSs+USs)

                CScmap = [map(int, (255*g,255*b,255*r)) for r,g,b,a in 
                    matplotlib.cm.winter(np.linspace(0,255,len(CSs)).astype(int))]
                UScmap = [map(int, (255*g,255*b,255*r)) for r,g,b,a in 
                    matplotlib.cm.gist_rainbow(np.linspace(0,255,len(USs)).astype(int))]
                Othercmap = [map(int, (255*g,255*b,255*r)) for r,g,b,a in 
                    matplotlib.cm.YlOrBr(np.linspace(0,255,len(others)).astype(int))]
                
                def eventMarks(Labels, cmaps, fish):
                    for label, cmap in zip(Labels, cmaps):
                        Events = self.EventData[fish][label]
                        for e in Events:
                            __x = int(w*e/self.nmax_frame)
                            cv2.line(frame, (__x, tB+0), (__x, tB+60), cmap, thickness=2)
                
                eventMarks(CSs, CScmap[::-1], fish)
                eventMarks(USs, UScmap[::-1], fish)
                eventMarks(others, Othercmap, fish)
           
            if SVy is not None: # z level
                cv2.polylines(frame, [np.array([t, tB+100*(SVy-tB)/tB]).astype(np.int32).T], False, color)

            # ringpixels
            cv2.polylines(frame, [np.array([t, 30+tB-30*rp/rp.max()]).astype(np.int32).T], False, color) # (0,255,100)

            # completion rate
            finished = 1*np.isnan(self.TVtracking[fish][:,0]) + 1*np.isnan(self.SVtracking[fish][:,1]) \
                       + 1*np.isnan(self.ringpixel[fish]) + 1*np.isnan(self.TVtracking[fish][:,3])
            
            cv2.polylines(frame, [np.array([t,tB-10
                +10*(1-np.convolve(np.ones(30)/30, finished,'same'))]).astype(np.int32).T], False, color2)
        
        # ringpolyTV or ringpolySV being edited.
        if self.ringpolyTV or self.ringpolySV:
            if 1<len(self.clicks)<8:
                cv2.polylines(frame, [np.array(self.clicks, dtype=np.int32)], False, color, thickness=2)
            elif len(self.clicks) == 8:
                params = cv2.fitEllipse(np.array(self.clicks))
                cv2.ellipse(frame, params, color=1, thickness=2)
                for x,y in self.clicks:
                    cv2.circle(frame, (x,y), 2, (100,222,100))
                cv2.circle(frame, (int(params[0][0]), int(params[0][1])), 2, (0,100,222))
        # ringpolyTV and/or ringpolySV ready
        if not self.ringpolyTV:
            if fish in self.ringpolyTVArray.keys():
                if self.ringpolyTVArray[fish] is not None:
                    _x,_y,_w,_h,_angle = self.ringpolyTVArray[fish][n,:]
                    cv2.ellipse(frame, ((_x,_y), (_w,_h), _angle), color, 2) # (0,85,200)
        if not self.ringpolySV:
            if fish in self.ringpolySVArray.keys():
                if self.ringpolySVArray[fish] is not None:
                    _x,_y,_w,_h,_angle = self.ringpolySVArray[fish][n,:]
                    cv2.ellipse(frame, ((_x,_y), (_w,_h), _angle), color, 2)
                    
                    # ringAppearochLevel
                    ringAppearochLevel = self.ringAppearochLevel.GetValue() + int(_y)
                    cv2.line(frame, (SVx1,ringAppearochLevel), (SVx2,ringAppearochLevel), color2)

        return frame


if __name__ == '__main__':
    app = wx.App(0)
    fishicon = wx.Icon('./resources/fish_multi.ico', wx.BITMAP_TYPE_ICO)
    myFrame = wxGui(pos=(0,40))

    # for debug
    # fp = r"R:\Data\itoiori\behav\adult whitlock\conditioning\NeuroD\Aug4\2015-08-04\Aug-04-2015_13_53_00.avi"
    # fp = r"R:\Data\itoiori\behav\adult whitlock\conditioning\NeuroD\Aug4\2015-08-04\Aug-04-2015_14_35_50.avi"
    # myFrame.LoadAvi(None, fp)
    # myFrame.targetfish.SetSelection(0)
    # myFrame.OnChoice(None)

    app.MainLoop()
    app.Destroy() # good to have this for sublimeREPL
