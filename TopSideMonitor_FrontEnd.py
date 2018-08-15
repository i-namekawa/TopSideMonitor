import datetime, getpass, os, subprocess, sys

import numpy as np
import wx
import wx.gizmos as gizmos

WhereIAm = os.getcwd()
from topsidecameras import twocameras  # this will change the current path
os.chdir(WhereIAm) # coming back

username = getpass.getuser()
defaultPath = os.path.join('C:\\Data', username)


class FileDrop(wx.FileDropTarget):
    def __init__(self, parent):
        wx.FileDropTarget.__init__(self)
        self.parent = parent
    def OnDropFiles(self, x, y, filenames):
        fp = filenames[0] # discard other files.
        self.parent.datafolder.SetValue( os.path.dirname(fp) )

class UserInput(wx.Frame):

    def __init__(self, parent):
        
        style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX
        wx.Frame.__init__(self, parent, size=(200,400), 
                          title='TopSideMonitor frontend',
                          style=style)
        fishicon = wx.Icon('./resources/fish_multi.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(fishicon)
        self.SetDropTarget(FileDrop(self))
        self.SetBackgroundColour('#EEEEEE')
        
        if os.path.exists(username+'.ini'):
            print 'ini file for %s found.' % username
            with open(username+'.ini','rt') as f:
                for line in f.readlines():
                    if line.count('=') == 1:
                        exec line
        else:
            datafolder = defaultPath
            textpos = "10,10"
            textcolor = "100"
            qscale = "3"
            textOn = 1
            videofps = '30'
            freeRun = 1
            camerafps = '30'
            maxFrame = '600'
            maxTime = str(int(maxFrame) / float(videofps))
        
        self.updateTooltip(None)

        self.SetToolTip(wx.ToolTip(self.tooltipmsg))
        
        self.folderdialog = wx.Button(self, -1, 'Data folder' )
        self.folderdialog.SetToolTip(wx.ToolTip('You can also drop a folder to the panel.'))
        self.datafolder = wx.TextCtrl(self, -1, datafolder, size=(260,-1), style=wx.TE_CENTER)
        self.datafolder.SetToolTip(wx.ToolTip(self.datafolder.GetValue()))
        
        self.textOn = wx.CheckBox(self, -1, 'Show text?' )
        self.textOn.SetValue(textOn)
        self.textpos = wx.TextCtrl(self, -1, textpos, size=(80, -1), style=wx.TE_RIGHT)
        self.textcolor = wx.TextCtrl(self, -1, textcolor, size=(80, -1), style=wx.TE_RIGHT)
        
        self.camerafpsId = wx.NewId()
        self.videofpsId = wx.NewId()
        self.camerafps = wx.SpinCtrl(self, self.camerafpsId, str(camerafps), size=(80, -1), min = 0, max = 202)
        self.videofps = wx.SpinCtrl(self, self.videofpsId, str(videofps), size=(80, -1), min = 0, max = 202)
        self.updatebtn = wx.Button(self, -1, 'Launch' )
        self.exitbtn = wx.Button(self, -1, 'Exit (ESC)' )
        self.folderbtn = wx.Button(self, -1, 'Open data folder' )
        
        self.freeRun = wx.CheckBox(self, -1, 'Free Run?' )
        self.freeRun.SetValue(freeRun)
        
        self.maxFrameId = wx.NewId()
        self.maxTimeId = wx.NewId()
        self.qscaleId = wx.NewId()
        self.maxFrame = wx.SpinCtrl(self, self.maxFrameId, maxFrame, size=(80, -1))
        self.maxFrame.Enable(self.freeRun.GetValue()==False)
        self.maxFrame.SetRange(1, 2**30)
        
        self.maxTime = wx.SpinCtrl(self, self.maxTimeId, maxTime, size=(80, -1))
        self.maxTime.Enable(self.freeRun.GetValue()==False)
        self.maxTime.SetRange(1, 2**30)

        self.qscale = wx.SpinCtrl(self, self.qscaleId, qscale, size=(80, -1))
        self.qscale.SetRange(2, 2000)
        
        self.lock = wx.ToggleButton(self, label='Lock')
        #self.lock.SetValue(True)

        self.gauge = wx.Gauge(self, -1, size=(0,15))
        self.gauge.Show(False)
        
        gbs = wx.GridBagSizer(6,4)
        gbs.Add(self.folderdialog, (0,0), flag=wx.ALIGN_CENTRE|wx.ALL, border=2)
        gbs.Add(self.datafolder, (0,1), span=(1,3), flag=wx.ALIGN_RIGHT|wx.ALL, border=2)
        
        gbs.Add( wx.StaticText(self, -1, 'Camera FPS' ), (4,0), flag=wx.ALIGN_CENTER )
        gbs.Add( self.camerafps, (4,1), flag=wx.ALIGN_RIGHT)
        gbs.Add( wx.StaticText(self, -1, 'Video FPS' ), (4,2), flag=wx.ALIGN_CENTER )
        gbs.Add( self.videofps, (4,3), flag=wx.ALIGN_LEFT)

        gbs.Add( self.textOn, (1,2) , span=(1,2), flag=wx.ALIGN_CENTER )
        gbs.Add( wx.StaticText(self, -1, 'Text 8-bit color' ), (2,2), flag=wx.ALIGN_CENTER )
        gbs.Add( self.textcolor, (2,3), flag=wx.ALIGN_LEFT)
        gbs.Add( wx.StaticText(self, -1, 'Text pos (x,y)' ), (3,2), flag=wx.ALIGN_CENTER )
        gbs.Add( self.textpos, (3,3), flag=wx.ALIGN_LEFT)
        
        gbs.Add( self.freeRun, (1,0), span=(1,2), flag=wx.ALIGN_CENTER )
        gbs.Add( wx.StaticText(self, -1, 'Max Frame' ), (2,0), flag=wx.ALIGN_CENTER )
        gbs.Add( self.maxFrame, (2,1), flag=wx.ALIGN_RIGHT )
        gbs.Add( wx.StaticText(self, -1, 'Max Time (sec)' ), (3,0), flag=wx.ALIGN_CENTER )
        gbs.Add( self.maxTime, (3,1), flag=wx.ALIGN_RIGHT )

        gbs.Add( self.lock, (5,1), flag=wx.ALIGN_CENTER )
        gbs.Add( wx.StaticText(self, -1, 'qscale (if 2-31)\nor kbits/s (>31)' ), (5,2), flag=wx.ALIGN_CENTER )
        gbs.Add( self.qscale, (5,3), flag=wx.ALIGN_LEFT )

        gbs.Add( self.updatebtn, (6,0), span=(1,1), flag=wx.ALIGN_CENTER )
        gbs.Add( self.folderbtn, (6,1), span=(1,2), flag=wx.ALIGN_CENTER )
        gbs.Add( self.exitbtn, (6,3), span=(1,1), flag=wx.ALIGN_CENTER )

        self.OnTextCheck(None)

        self.LED = gizmos.LEDNumberCtrl(self, -1, (25,100), (280, 50))
        self.LED.SetAlignment(gizmos.LED_ALIGN_RIGHT)
        self.LED.SetDrawFaded(True)
        self.LED.SetValue("0")
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add( self.gauge, 1, wx.EXPAND|wx.ALL, 1 )
        vbox.Add( gbs, 0, wx.ALIGN_CENTRE|wx.ALL, 5 )
        vbox.Add( self.LED, 0, wx.EXPAND|wx.ALIGN_CENTRE|wx.ALL, 5 )
        
        self.SetSizerAndFit(vbox)
        self.Show()
        
        self.folderdialog.Bind(wx.EVT_BUTTON, self.OnFolderDir)
        self.textOn.Bind(wx.EVT_CHECKBOX, self.OnTextCheck)
        self.freeRun.Bind(wx.EVT_CHECKBOX, self.OnTextCheck)
        self.exitbtn.Bind(wx.EVT_BUTTON, self.OnExit)
        self.folderbtn.Bind(wx.EVT_BUTTON, self.OnDataFolder)
        self.updatebtn.Bind(wx.EVT_BUTTON, self.OnUpdate)
        self.lock.Bind(wx.EVT_TOGGLEBUTTON, self.OnLock)
        self.maxTime.Bind(wx.EVT_SPIN, self.OnSpins)
        self.maxFrame.Bind(wx.EVT_SPIN, self.OnSpins)
        self.camerafps.Bind(wx.EVT_SPIN, self.OnSpins)
        self.videofps.Bind(wx.EVT_SPIN, self.OnSpins)
        
        self.Bind(wx.EVT_KEY_UP, self.OnExit)
        
        self.timer = wx.Timer(self) # Timer for updating tooltip
        self.Bind(wx.EVT_TIMER, self.updateTooltip, self.timer)
        self.timer.Start(1000*60) # every hour in ms

        self.recording = False
    

    def updateTooltip(self, event):
        today = datetime.date.today().strftime('%Y-%m-%d')
        self.tooltipmsg = 'Today is ' + today
        print self.tooltipmsg
    
    def OnDataFolder(self, event):
        folder = self.datafolder.GetValue()
        subprocess.Popen(r'explorer %s' % folder)
    
    def OnLock(self, event):
        if not self.lock.GetValue():
            print 'Not locked'
            self.lock.SetBackgroundColour('#EEEEEE')
        else:
            print 'locked'
            self.lock.SetBackgroundColour('#EE2222')
        
    def OnSpins(self, event):
        _id = event.GetId()
        #fps = self.videofps.GetValue()
        fps = self.camerafps.GetValue()
        if _id == self.maxFrameId:
            framenum = self.maxFrame.GetValue()
            self.maxTime.SetValue(framenum / fps)
        elif _id == self.maxTimeId:
            _time = self.maxTime.GetValue()
            self.maxFrame.SetValue(_time * fps)
        else:
            _camerafps = float( self.camerafps.GetValue() )
            _videofps = float( self.videofps.GetValue() )
            #self.skipframe.SetValue('%2.2f'%(_camerafps/_videofps))
        
    def OnTextCheck(self, event):
        flag1 = self.textOn.GetValue()
        self.textpos.Enable(flag1)
        self.textcolor.Enable(flag1)
        flag2 = self.freeRun.GetValue() == False
        self.maxFrame.Enable(flag2)
        self.maxTime.Enable(flag2)
    
    def OnFolderDir(self, event):
        dlg = wx.DirDialog(self, 
                        message="Choose the data folder",
                        defaultPath=self.datafolder.GetValue()
                        )
        if dlg.ShowModal() == wx.ID_OK:
            self.datafolder.SetValue( dlg.GetPath() )
        self.datafolder.SetToolTip(wx.ToolTip(self.datafolder.GetValue()))
        dlg.Destroy()

    def OnUpdate(self, event):
        if not self.recording:
            datafolder = self.datafolder.GetValue()
            if not os.path.exists(datafolder):
                os.mkdir(datafolder)
            
            videofps = self.videofps.GetValue()
            camerafps = self.camerafps.GetValue()
            textpos = tuple([int(aa) for aa in self.textpos.GetValue().split(',')])
            textcolor = int(self.textcolor.GetValue())
            textOn = self.textOn.GetValue()
            qscale = self.qscale.GetValue()
            if self.freeRun.GetValue():
                maxFrame = None
                maxTime = None
            else:
                maxFrame = self.maxFrame.GetValue()
                maxTime = maxFrame / camerafps
            
            self.recording = True
            
            style=wx.STAY_ON_TOP
            self.gauge.Show(True)
            self.SetWindowStyle(style)
            self.Refresh()
            
            camera = twocameras( camerafps, videofps, datafolder, 
                                 maxTime, qscale, textOn, textcolor, textpos, self )
            
            self.recording = False
            self.gauge.Show(False)
            style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX
            self.SetWindowStyle(style)

            if not camera:
                msg = 'No camera found on your system!'
                dlg = wx.MessageDialog(None, msg, style=wx.OK) # None for top level
                if dlg.ShowModal() == wx.ID_OK:
                    pass
                dlg.Destroy()
        
    def OnExit(self, event):
        
        event.Skip() # important
        if not event.IsCommandEvent():
            if event.GetKeyCode() != 27:  # esc
                return
        
        with open(username+'.ini','wt') as f:
            f.write( 'datafolder = r\'%s\'\n' % self.datafolder.GetValue() )
            f.write( 'videofps = %d\n' % self.videofps.GetValue() )
            f.write( 'camerafps = %d\n' % self.camerafps.GetValue() )
            f.write( 'textpos = \'%s\'\n' % self.textpos.GetValue() )
            f.write( 'textcolor = \'%s\'\n' % self.textcolor.GetValue() )
            f.write( 'textOn = %d\n' % self.textOn.GetValue() )
            f.write( 'freeRun = %d\n' % self.freeRun.GetValue() )
            f.write( 'maxFrame = \'%s\'\n' % self.maxFrame.GetValue() )
            f.write( 'maxTime = \'%s\'\n' % self.maxTime.GetValue() )
            f.write( 'qscale = \'%s\'\n' % self.qscale.GetValue() )
        
        self.Destroy()
    
if __name__ == '__main__':
    app = wx.App(0)
    frame = UserInput(None)
    app.MainLoop()
