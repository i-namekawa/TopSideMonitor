import os, sys, time
from glob import glob

import cv2

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['figure.facecolor'] = 'w'

from scipy.signal import argrelextrema
import scipy.stats as stats
import scipy.io as sio
from scipy import signal

from xlwt import Workbook

# specify these in mm to match your behavior chamber.
CHMAMBER_LENGTH=235
WATER_HIGHT=40


# quick plot should also show xy_within and location_one_third etc
# summary PDF: handle exception when a pickle file missing some fish in other pickle file


## these three taken from http://stackoverflow.com/a/18420730/566035
def strided_sliding_std_dev(data, radius=5):
    windowed = rolling_window(data, (2*radius, 2*radius))
    shape = windowed.shape
    windowed = windowed.reshape(shape[0], shape[1], -1)
    return windowed.std(axis=-1)

def rolling_window(a, window):
    """Takes a numpy array *a* and a sequence of (or single) *window* lengths
    and returns a view of *a* that represents a moving window."""
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a

def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
       raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
## stealing ends here... //




def filterheadxy(headx,heady,thrs_denom=10):
    
    b, a = signal.butter(8, 0.125)
    dhy = np.abs(np.hstack((0, np.diff(heady,1))))
    thrs = np.nanstd(dhy)/thrs_denom
    ind2remove = dhy>thrs
    headx[ind2remove] = np.nan
    heady[ind2remove] = np.nan

    headx = interp_nan(headx)
    heady = interp_nan(heady)
    headx = signal.filtfilt(b, a, headx, padlen=150)
    heady = signal.filtfilt(b, a, heady, padlen=150)

    return headx,heady

def smoothRad(theta, thrs=np.pi/4*3):
    jumps = (np.diff(theta) > thrs).nonzero()[0]
    print 'jumps.size', jumps.size
    while jumps.size:
        # print '%d/%d' % (jumps[0], theta.size)
        theta[jumps+1] -= np.pi
        jumps = (np.diff(theta) > thrs).nonzero()[0]

    return theta

def datadct2array(data, key1, key2):

    # put these in a MATLAB CELL
    trialN = len(data[key1][key2])
    matchedUSnameP = np.zeros((trialN,), dtype=np.object)
    fnameP = np.zeros((trialN,), dtype=np.object)
    # others to append to a list
    eventsP = []
    speed3DP = []
    movingSTDP = []
    d2inflowP = []
    xP, yP, zP = [], [], []
    XP, YP, ZP = [], [], []
    ringpixelsP = []
    peaks_withinP = []
    swimdir_withinP = []
    xy_withinP = []
    location_one_thirdP = []
    dtheta_shapeP = []
    dtheta_velP = []
    turns_shapeP = []
    turns_velP = []

    for n, dct in enumerate(data[key1][key2]):
        # MATLAB CELL
        matchedUSnameP[n] = dct['matchedUSname']
        fnameP[n] = dct['fname']

        # 2D array
        eventsP.append([ele if type(ele) is not list else ele[0] for ele in dct['events']])
        speed3DP.append(dct['speed3D'])
        movingSTDP.append(dct['movingSTD'])
        d2inflowP.append(dct['d2inflow'])
        xP.append(dct['x'])
        yP.append(dct['y'])
        zP.append(dct['z'])
        XP.append(dct['X'])
        YP.append(dct['Y'])
        ZP.append(dct['Z'])
        ringpixelsP.append(dct['ringpixels'])
        peaks_withinP.append(dct['peaks_within'])
        swimdir_withinP.append(dct['swimdir_within'])
        xy_withinP.append(dct['xy_within'])
        location_one_thirdP.append(dct['location_one_third'])
        dtheta_shapeP.append(dct['dtheta_shape'])
        dtheta_velP.append(dct['dtheta_vel'])
        turns_shapeP.append(dct['turns_shape'])
        turns_velP.append(dct['turns_vel'])

    TVroi = np.array(dct['TVroi'])
    SVroi = np.array(dct['SVroi'])

    return matchedUSnameP, fnameP, np.array(eventsP), np.array(speed3DP), np.array(d2inflowP), \
            np.array(xP), np.array(yP), np.array(zP), np.array(XP), np.array(YP), np.array(ZP), \
            np.array(ringpixelsP), np.array(peaks_withinP), np.array(swimdir_withinP), \
            np.array(xy_withinP), np.array(dtheta_shapeP), np.array(dtheta_velP), \
            np.array(turns_shapeP), np.array(turns_velP), TVroi, SVroi

def pickle2mat(fp, data=None):
    # fp : full path to pickle file
    # data : option to provide data to skip np.load(fp) 
    if not data:
        data = np.load(fp)

    for key1 in data.keys():
        for key2 in data[key1].keys():

            matchedUSname, fname, events, speed3D, d2inflow, x, y, z, X, Y, Z, \
            ringpixels, peaks_within, swimdir_within, xy_within, dtheta_shape, dtheta_vel, \
            turns_shape, turns_vel, TVroi, SVroi = datadct2array(data, key1, key2)

            datadict = {
                'matchedUSname' : matchedUSname,
                'fname' : fname,
                'events' : events,
                'speed3D' : speed3D,
                'd2inflow' : d2inflow,
                'x' : x,
                'y' : y,
                'z' : z,
                'X' : X,
                'Y' : Y,
                'Z' : Z,
                'ringpixels' : ringpixels,
                'peaks_within' : peaks_within,
                'swimdir_within' : swimdir_within,
                'xy_within' : xy_within,

               'dtheta_shape' : dtheta_shape,
               'dtheta_vel' : dtheta_vel,
               'turns_shape' : turns_shape,
               'turns_vel' : turns_vel,

                'TVroi' : TVroi,
                'SVroi' : SVroi,
            }
            outfp = '%s_%s_%s.mat' % (fp[:-7],key1,key2)
            sio.savemat(outfp, datadict, oned_as='row', do_compression=True)

def interp_nan(x):
    '''
    Replace nan by interporation
    http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    '''
    ok = -np.isnan(x)
    if (ok == False).all():
        return x
    else:
        xp = ok.ravel().nonzero()[0]
        fp = x[ok]
        _x = np.isnan(x).ravel().nonzero()[0]
        x[-ok] = np.interp(_x, xp, fp)
        return x

def polytest(x,y,rx,ry,rw,rh,rang):
    points=cv2.ellipse2Poly(
            (rx,ry),
            axes=(rw/2,rh/2),
            angle=rang,
            arcStart=0,
            arcEnd=360,
            delta=3
        )
    return cv2.pointPolygonTest(np.array(points), (x,y), measureDist=1)

def depthCorrection(z,x,TVx1,TVx2,SVy1,SVy2,SVy3):
    z0 = z - SVy1
    x0 = x - TVx1
    mid = (SVy2-SVy1)/2
    adj = (z0 - mid) / (SVy2-SVy1) * (SVy2-SVy3) * (1-(x0)/float(TVx2-TVx1))
    return z0 + adj + SVy1 # back to abs coord

def putNp2xls(array, ws):
    for r, row in enumerate(array):
        for c, val in enumerate(row):
            ws.write(r, c, val)

def drawLines(mi, ma, events, fps=30.0):

    CS, USs, preRange = events

    plot([CS-preRange, CS-preRange], [mi,ma], '--c')                    # 2 min prior odor
    plot([CS         , CS         ], [mi,ma], '--g', linewidth=2)       # CS onset
    if USs:
        if len(USs) > 3:
            colors = 'r' * len(USs)
        else:
            colors = [_ for _ in ['r','b','c'][:len(USs)]]
        for c,us in zip(colors, USs):
            plot([us, us],[mi,ma], linestyle='--', color=c, linewidth=2)            # US onset
        plot([USs[0]+preRange/2,USs[0]+preRange/2], [mi,ma], linestyle='--', color=c, linewidth=2)    # end of US window
        xtck = np.arange(0, max(CS+preRange, max(USs)), 0.5*60*fps) # every 0.5 min tick
    else:
        xtck = np.arange(0, CS+preRange, 0.5*60*fps) # every 0.5 min tick
    
    xticks(xtck, xtck/fps/60)
    gca().xaxis.set_minor_locator(MultipleLocator(5*fps)) # 5 s minor ticks


def approachevents(x,y,z, ringpolyTVArray, ringpolySVArray, fishlength=134, thrs=None):
    '''
    fishlength: some old scrits may call this with fishlength
    thrs: multitrack GUI provides this by ringAppearochLevel spin control. 
          can be an numpy array (water level change etc)
    '''

    smoothedz = np.convolve(np.hanning(10)/np.hanning(10).sum(), z, 'same')
    peaks = argrelextrema(smoothedz, np.less)[0] # less because 0 is top in image.

    # now filter peaks by height.
    ringLevel = ringpolySVArray[:,1]
    if thrs is None:
        thrs = ringLevel+fishlength/2

    if type(thrs) == int:  # can be numpy array or int
        thrs = ringLevel.mean() + thrs
        peaks = peaks[ z[peaks] < thrs ]
    else: # numpy array should be ready to use
        peaks = peaks[ z[peaks] < thrs[peaks] ]

    # now filter out by TVringCenter
    peaks_within = get_withinring(ringpolyTVArray, peaks, x, y)

    return smoothedz, peaks_within


def get_withinring(ringpolyTVArray, timepoints, x, y):
    
    rx = ringpolyTVArray[:,0].astype(np.int)
    ry = ringpolyTVArray[:,1].astype(np.int)
    rw = ringpolyTVArray[:,2].astype(np.int)
    rh = ringpolyTVArray[:,3].astype(np.int)
    rang = ringpolyTVArray[:,4].astype(np.int)
    # poly test
    peaks_within = []
    for p in timepoints:
        points=cv2.ellipse2Poly(
                (rx[p],ry[p]),
                axes=(rw[p]/2,rh[p]/2),
                angle=rang[p],
                arcStart=0,
                arcEnd=360,
                delta=3
            )
        inout = cv2.pointPolygonTest(np.array(points), (x[p],y[p]), measureDist=1)
        if inout > 0:
            peaks_within.append(p)

    return peaks_within


def location_ring(x,y,ringpolyTVArray):

    rx = ringpolyTVArray[:,0].astype(np.int)
    ry = ringpolyTVArray[:,1].astype(np.int)
    rw = ringpolyTVArray[:,2].astype(np.int)
    rh = ringpolyTVArray[:,3].astype(np.int)
    d2ringcenter = np.sqrt((x-rx)**2 + (y-ry)**2)

    # filter by radius 20% buffer in case the ring moves around
    indices = (d2ringcenter < 1.2*max(rw.max(), rh.max())).nonzero()[0] 
    xy_within = get_withinring(ringpolyTVArray, indices, x, y)
  
    return xy_within


def swimdir_analysis(x,y,z,ringpolyTVArray,ringpolySVArray,TVx1,TVy1,TVx2,TVy2,fps=30.0):
    # smoothing
    # z = np.convolve(np.hanning(16)/np.hanning(16).sum(), z, 'same')

    # two cameras have different zoom settings. So, distance per pixel is different. But, for
    # swim direction, it does not matter how much x,y are compressed relative to z.

    # ring z level from SV
    rz = ringpolySVArray[:,1].astype(np.int)
    # ring all other params from TV
    rx = ringpolyTVArray[:,0].astype(np.int)
    ry = ringpolyTVArray[:,1].astype(np.int)
    rw = ringpolyTVArray[:,2].astype(np.int)
    rh = ringpolyTVArray[:,3].astype(np.int)
    rang = ringpolyTVArray[:,4].astype(np.int)

    speed3D = np.sqrt( np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2 )
    speed3D = np.hstack(([0], speed3D))

    # line in 3D http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfLines.aspx
    # x-x0   y-y0   z-z0
    # ---- = ---- = ----
    #   a      b      c
    # solve them for z = rz. x0,y0,z0 are tvx, tvy, svy
    # x = (a * (rz-z)) / c + x0

    dt = 3 # define slope as diff between current and dt frame before
    a = np.hstack( (np.ones(dt), x[dt:]-x[:-dt]) )
    b = np.hstack( (np.ones(dt), y[dt:]-y[:-dt]) )
    c = np.hstack( (np.ones(dt), z[dt:]-z[:-dt]) )
    c[c==0] = np.nan # avoid zero division

    water_x = (a * (rz-z) / c) + x
    water_y = (b * (rz-z) / c) + y

    upwards = c<-2/30.0*fps # not accurate when c is small or negative
    xok = (TVx1 < water_x) & (water_x < TVx2)
    yok = (TVy1 < water_y) & (water_y < TVy2)

    filtered = upwards & xok & yok# & -np.isinf(water_x) & -np.isinf(water_y)

    water_x[-filtered] = np.nan
    water_y[-filtered] = np.nan
    
    # figure()
    # ax = subplot(111)
    # ax.imshow(npData['TVbg'], cmap=cm.gray) # clip out from TVx1,TVy1
    # ax.plot(x-TVx1, y-TVy1, 'c')
    # ax.plot(water_x-TVx1, water_y-TVy1, 'r.')
    # xlim([0, TVx2-TVx1]); ylim([TVy2-TVy1, 0])
    # draw(); show()

    SwimDir = []
    for n in filtered.nonzero()[0]:
        inout = polytest(water_x[n],water_y[n],rx[n],ry[n],rw[n],rh[n],rang[n])
        SwimDir.append((n, inout, speed3D[n])) # inout>0 are inside

    return SwimDir, water_x, water_y

def plot_eachTr(events, x, y, z, inflowpos, ringpixels, peaks_within, swimdir_within=None, 
    pp=None, _title=None, fps=30.0, inmm=False):

    CS, USs, preRange = events
    # preRange = 3600 2 min prior and 1 min after CS. +900 for 0.5 min
    if USs:
        xmin, xmax = CS-preRange-10*fps, USs[0]+preRange/2+10*fps
    else:
        xmin, xmax = CS-preRange-10*fps, CS+preRange/2+(23+10)*fps

    fig = figure(figsize=(12,8), facecolor='w')
    
    subplot(511)        # Swimming speed
    speed3D = np.sqrt( np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2 )
    drawLines(np.nanmin(speed3D), np.nanmax(speed3D), events, fps) # go behind
    plot(speed3D)

    movingSTD = np.append( np.zeros(fps*10), strided_sliding_std_dev(speed3D, fps*10) )
    plot(movingSTD, linewidth=2)

    plot(np.ones_like(speed3D) * speed3D.std()*6, '-.', color='gray')
    ylim([-5, speed3D[xmin:xmax].max()])
    xlim([xmin,xmax]); title(_title)
    if inmm:
        ylabel('Speed 3D (mm),\n6SD thr'); 
    else:
        ylabel('Speed 3D, 6SD thr'); 
    
    ax = subplot(512)   # z level
    drawLines(z.min(), z.max(), events)
    plot(z, 'b')
    
    pkx = peaks_within.nonzero()[0]
    if inmm:
        plot(pkx, peaks_within[pkx]*z[xmin:xmax].max()*0.97, 'mo')
        if swimdir_within is not None:
            ___x = swimdir_within.nonzero()[0]
            plot(___x, swimdir_within[___x]*z[xmin:xmax].max()*0.96, 'g+')
        ylim([z[xmin:xmax].min()*0.95, z[xmin:xmax].max()])
        xlim([xmin,xmax]); ylabel('Z (mm)')
    else:
        plot(pkx, peaks_within[pkx]*z[xmin:xmax].min()*0.97, 'mo')
        if swimdir_within is not None:
            ___x = swimdir_within.nonzero()[0]
            plot(___x, swimdir_within[___x]*z[xmin:xmax].min()*0.96, 'g+')
        ylim([z[xmin:xmax].min()*0.95, z[xmin:xmax].max()])
        ax.invert_yaxis(); xlim([xmin,xmax]); ylabel('z')
    
    subplot(513)        # x
    drawLines(x.min(), x.max(), events)
    plot(x, 'b')
    plot(y, 'g')
    xlim([xmin,xmax]); ylabel('x,y')
    
    subplot(514)        # Distance to the inflow tube
    xin, yin, zin = inflowpos
    d2inflow = np.sqrt((x-xin) ** 2 + (y-yin) ** 2 + (z-zin) ** 2 )
    drawLines(d2inflow.min(), d2inflow.max(), events)
    plot(d2inflow)
    ylim([d2inflow[xmin:xmax].min(), d2inflow[xmin:xmax].max()])
    xlim([xmin,xmax]); ylabel('distance to\ninflow tube')
    
    subplot(515)        # ringpixels: it seems i never considered TV x,y for this
    rpmax, rpmin = np.nanmax(ringpixels[xmin:xmax]), np.nanmin(ringpixels[xmin:xmax])
    drawLines(rpmin, rpmax, events)
    plot(ringpixels)
    plot(pkx, peaks_within[pkx]*rpmax*1.06, 'mo')
    if swimdir_within is not None:
        plot(___x, swimdir_within[___x]*rpmax*1.15, 'g+')
    ylim([-100, rpmax*1.2])
    xlim([xmin,xmax]); ylabel('ringpixels')

    tight_layout()

    if pp:
        fig.savefig(pp, format='pdf')

    rng = np.arange(CS-preRange, CS+preRange, dtype=np.int)

    return speed3D[rng], movingSTD[rng], d2inflow[rng], ringpixels[rng]

def plot_turnrates(events, dthetasum_shape,dthetasum_vel,turns_shape,turns_vel, 
            pp=None, _title=None, thrs=np.pi/4*(133.33333333333334/120), fps=30.0):

    CS, USs, preRange = events
    # preRange = 3600 2 min prior and 1 min after CS. +900 for 0.5 min
    if USs:
        xmin, xmax = CS-preRange-10*fps, USs[0]+preRange/2+10*fps
    else:
        xmin, xmax = CS-preRange-10*fps, CS+preRange/2+(23+10)*fps

    fig = figure(figsize=(12,8), facecolor='w')

    subplot(211)
    drawLines(dthetasum_shape.min(), dthetasum_shape.max(), events)
    plot(np.ones_like(dthetasum_shape)*thrs,'gray',linestyle='--')
    plot(-np.ones_like(dthetasum_shape)*thrs,'gray',linestyle='--')
    plot(dthetasum_shape)
    dmax = dthetasum_shape[xmin:xmax].max()
    plot(turns_shape, (0.5+dmax)*np.ones_like(turns_shape), 'o')
    temp = np.zeros_like(dthetasum_shape)
    temp[turns_shape] = 1
    shape_cumsum = np.cumsum(temp)
    shape_cumsum -= shape_cumsum[xmin]
    plot( shape_cumsum / shape_cumsum[xmax] * (dmax-dthetasum_shape.min()) + dthetasum_shape.min())
    xlim([xmin,xmax]); ylabel('Shape based'); title('Orientation change per 4 frames: ' + _title)
    ylim([dthetasum_shape[xmin:xmax].min()-1, dmax+1])

    subplot(212)
    drawLines(dthetasum_vel.min(), dthetasum_vel.max(), events)
    plot(np.ones_like(dthetasum_vel)*thrs,'gray',linestyle='--')
    plot(-np.ones_like(dthetasum_vel)*thrs,'gray',linestyle='--')
    plot(dthetasum_vel)
    dmax = dthetasum_vel[xmin:xmax].max()
    plot(turns_vel, (0.5+dmax)*np.ones_like(turns_vel), 'o')
    temp = np.zeros_like(dthetasum_vel)
    temp[turns_vel] = 1
    vel_cumsum = np.cumsum(temp)
    vel_cumsum -= vel_cumsum[xmin]
    plot( vel_cumsum / vel_cumsum[xmax] * (dmax-dthetasum_shape.min()) + dthetasum_shape.min())

    ylim([dthetasum_vel[xmin:xmax].min()-1, dmax+1])
    xlim([xmin,xmax]); ylabel('Velocity based')

    tight_layout()

    if pp:
        fig.savefig(pp, format='pdf')

def trajectory(x, y, z, rng, ax, _xlim=[0,640], _ylim=[480,480+300], _zlim=[150,340], 
                    color='b', fps=30.0, ringpolygon=None):
    ax.plot(x[rng],y[rng],z[rng], color=color)
    ax.view_init(azim=-75, elev=-180+15)
    if ringpolygon:
        rx, ry, rz = ringpolygon
        ax.plot(rx, ry, rz, color='gray')
    ax.set_xlim(_xlim[0],_xlim[1])
    ax.set_ylim(_ylim[0],_ylim[1])
    ax.set_zlim(_zlim[0],_zlim[1])
    title(("(%2.1f min to %2.1f min)" % (rng[0]/fps/60.0,(rng[-1]+1)/60.0/fps)))
    draw()

def plotTrajectory(x, y, z, events, _xlim=None, _ylim=None, _zlim=None, fps=30.0, pp=None, ringpolygon=None):
    CS, USs, preRange = events
    rng1 = np.arange(CS-preRange, CS-preRange/2, dtype=int)
    rng2 = np.arange(CS-preRange/2, CS, dtype=int)
    if USs:
        rng3 = np.arange(CS, min(USs), dtype=int)
        rng4 = np.arange(min(USs), min(USs)+preRange/2, dtype=int)
        combined = np.hstack((rng1,rng2,rng3,rng4))
    else:
        combined = np.hstack((rng1,rng2))

    if _xlim is None:
        _xlim = map( int, ( x[combined].min(), x[combined].max() ) )
    if _ylim is None:
        _ylim = map( int, ( y[combined].min(), y[combined].max() ) )
    if _zlim is None:
        _zlim = map( int, ( z[combined].min(), z[combined].max() ) )
        if ringpolygon:
            _zlim[0] = min( _zlim[0], int(ringpolygon[2][0]) )
    
    fig3D = plt.figure(figsize=(12,8), facecolor='w')
    ax = fig3D.add_subplot(221, projection='3d'); trajectory(x,y,z,rng1,ax,_xlim,_ylim,_zlim,'c',fps,ringpolygon)
    ax = fig3D.add_subplot(222, projection='3d'); trajectory(x,y,z,rng2,ax,_xlim,_ylim,_zlim,'c',fps,ringpolygon)
    if USs:
        ax = fig3D.add_subplot(223, projection='3d'); trajectory(x,y,z,rng3,ax,_xlim,_ylim,_zlim,'g',fps,ringpolygon)
        ax = fig3D.add_subplot(224, projection='3d'); trajectory(x,y,z,rng4,ax,_xlim,_ylim,_zlim,'r',fps,ringpolygon)
    tight_layout()

    if pp:
        fig3D.savefig(pp, format='pdf')

def add2DataAndPlot(fp, fish, data, createPDF):

    if createPDF:
        pp = PdfPages(fp[:-7]+'_'+fish+'.pdf')
    else:
        pp = None
    
    params = np.load(fp)
    fname = os.path.basename(fp).split('.')[0] + '.avi'
    dirname = os.path.dirname(fp)
    preRange = params[(fname, 'mog')]['preRange']
    fps = params[(fname, 'mog')]['fps']
    TVx1 = params[(fname, fish)]['TVx1']
    TVy1 = params[(fname, fish)]['TVy1']
    TVx2 = params[(fname, fish)]['TVx2']
    TVy2 = params[(fname, fish)]['TVy2']

    SVx1 = params[(fname, fish)]['SVx1']
    SVx2 = params[(fname, fish)]['SVx2']
    SVx3 = params[(fname, fish)]['SVx3']
    SVy1 = params[(fname, fish)]['SVy1']
    SVy2 = params[(fname, fish)]['SVy2']
    SVy3 = params[(fname, fish)]['SVy3']
    ringAppearochLevel = params[(fname, fish)]['ringAppearochLevel']

    _npz = os.path.join(dirname, os.path.join('%s_%s.npz' % (fname[:-4], fish)))
    # if os.path.exists(_npz):
    npData = np.load(_npz)
    tvx = npData['TVtracking'][:,0] # x with nan
    tvy = npData['TVtracking'][:,1] # y 
    headx = npData['TVtracking'][:,3] # headx 
    heady = npData['TVtracking'][:,4] # heady 
    svy = npData['SVtracking'][:,1] # z 
    InflowTubeTVArray = npData['InflowTubeTVArray']
    InflowTubeSVArray = npData['InflowTubeSVArray']
    inflowpos = InflowTubeTVArray[:,0], InflowTubeTVArray[:,1], InflowTubeSVArray[:,1]
    ringpixels = npData['ringpixel']
    ringpolyTVArray = npData['ringpolyTVArray']
    ringpolySVArray = npData['ringpolySVArray']
    TVbg = npData['TVbg']
    print os.path.basename(_npz), 'loaded.'

    x,y,z = map(interp_nan, [tvx,tvy,svy])

    # z level correction by depth (x)
    z = depthCorrection(z,x,TVx1,TVx2,SVy1,SVy2,SVy3)

    smoothedz, peaks_within = approachevents(x, y, z, 
        ringpolyTVArray, ringpolySVArray, thrs=ringAppearochLevel)
    # convert to numpy array from list 
    temp = np.zeros_like(x)
    temp[peaks_within] = 1
    peaks_within = temp

    # normalize to mm
    longaxis = float(max((TVx2-TVx1), (TVy2-TVy1))) # before rotation H is applied they are orthogonal
    waterlevel = float(SVy2-SVy1)
    X = (x-TVx1) / longaxis * CHMAMBER_LENGTH
    Y = (TVy2-y) / longaxis * CHMAMBER_LENGTH
    Z = (SVy2-z) / waterlevel * WATER_HIGHT  # bottom of chamber = 0, higher more positive 
    inflowpos_mm = ((inflowpos[0]-TVx1) / longaxis * CHMAMBER_LENGTH,
                    (TVy2-inflowpos[1]) / longaxis * CHMAMBER_LENGTH,
                    (SVy2-inflowpos[2]) / waterlevel * WATER_HIGHT )

    # do the swim direction analysis here
    swimdir, water_x, water_y = swimdir_analysis(x,y,z,
                ringpolyTVArray,ringpolySVArray,TVx1,TVy1,TVx2,TVy2,fps)
    # all of swimdir are within ROI (frame#, inout, speed) but not necessary within ring
    sdir = np.array(swimdir)
    withinRing = sdir[:,1]>0 # inout>0 are inside ring
    temp = np.zeros_like(x)
    temp[ sdir[withinRing,0].astype(int) ] = 1
    swimdir_within = temp

    # location_ring
    xy_within = location_ring(x,y, ringpolyTVArray)
    temp = np.zeros_like(x)
    temp[xy_within] = 1
    xy_within = temp

    # location_one_third
    if (TVx2-TVx1) > (TVy2-TVy1):
        if np.abs(np.arange(TVx1, longaxis+TVx1, longaxis/3) + longaxis/6 - inflowpos[0].mean()).argmin() == 2:
            location_one_third = x-TVx1 > longaxis/3*2
        else:
            location_one_third = x < longaxis/3
    else:
        if np.abs(np.arange(TVy1, longaxis+TVy1, longaxis/3) + longaxis/6 - inflowpos[1].mean()).argmin() == 2:
            location_one_third = y-TVy1 > longaxis/3*2
        else:
            location_one_third = y < longaxis/3

    # turn rate analysis (shape based)
    heady, headx = map(interp_nan, [heady, headx])
    headx, heady = filterheadxy(headx, heady)
    dy = heady - y
    dx = headx - x
    theta_shape = np.arctan2(dy, dx)

    # velocity based
    cx, cy = filterheadxy(x.copy(), y.copy())  # centroid x,y

    vx = np.append(0, np.diff(cx))
    vy = np.append(0, np.diff(cy))
    theta_vel = np.arctan2(vy, vx)

    # prepare ringpolygon for trajectory plot
    rx, ry, rw, rh, rang = ringpolyTVArray.mean(axis=0).astype(int) # use mm ver above
    rz = ringpolySVArray.mean(axis=0)[1].astype(int)
    
    RX = (rx-TVx1) / longaxis * CHMAMBER_LENGTH
    RY = (TVy2-ry) / longaxis * CHMAMBER_LENGTH
    RW = rw / longaxis * CHMAMBER_LENGTH / 2
    RH = rh / longaxis * CHMAMBER_LENGTH / 2
    RZ = (SVy2-rz) / waterlevel * WATER_HIGHT
    points = cv2.ellipse2Poly(
            (RX.astype(int),RY.astype(int)),
            axes=(RW.astype(int),RH.astype(int)),
            angle=rang,
            arcStart=0,
            arcEnd=360,
            delta=3
        )
    ringpolygon = [points[:,0], points[:,1], np.ones(points.shape[0]) * RZ]

    eventTypeKeys = params[(fname, fish)]['EventData'].keys()
    CSs = [_ for _ in eventTypeKeys if _.startswith('CS')]
    USs = [_ for _ in eventTypeKeys if _.startswith('US')]
    # print CSs, USs

    # events
    for CS in CSs:
        CS_Timings = params[(fname, fish)]['EventData'][CS]
        CS_Timings.sort()
        # initialize when needed
        if CS not in data[fish].keys():
            data[fish][CS] = []

        # now look around for US after it within preRange
        for t in CS_Timings:
            tr = len(data[fish][CS])+1
            rng = np.arange(t-preRange, t+preRange, dtype=np.int)

            matchedUSname = None
            for us in USs:
                us_Timings = params[(fname, fish)]['EventData'][us]
                matched = [_ for _ in us_Timings if t-preRange < _ < t+preRange]
                if matched:
                    events = [t, matched, preRange] # ex. CS+
                    matchedUSname = us
                    break
                else:
                    continue

            _title = '(%s, %s) trial#%02d %s (%s)' % (CS, matchedUSname[0], tr, fname, fish)
            print _title, events

            _speed3D, _movingSTD, _d2inflow, _ringpixels = plot_eachTr(events, X, Y, Z, inflowpos_mm, 
                                    ringpixels, peaks_within, swimdir_within, pp, _title, fps, inmm=True)
            
            # 3d trajectory
            _xlim = (0, CHMAMBER_LENGTH)
            _zlim = (RZ.max(),0)
            plotTrajectory(X, Y, Z, events, _xlim=_xlim, _zlim=_zlim, fps=fps, pp=pp, ringpolygon=ringpolygon)
            

            # turn rate analysis
            # shape based
            theta_shape[rng] = smoothRad(theta_shape[rng].copy(), thrs=np.pi/2)
            dtheta_shape = np.append(0, np.diff(theta_shape)) # full length

            kernel = np.ones(4)
            dthetasum_shape = np.convolve(dtheta_shape, kernel, 'same')
            
            # 4 frames = 1000/30.0*4 = 133.3 ms 
            thrs = (np.pi / 2) * (133.33333333333334/120) # Braubach et al 2009 90 degree in 120 ms 
            peaks_shape = argrelextrema(abs(dthetasum_shape), np.greater)[0]
            turns_shape = peaks_shape[ (abs(dthetasum_shape[peaks_shape]) > thrs).nonzero()[0] ]
            
            # velocity based
            theta_vel[rng] = smoothRad(theta_vel[rng].copy(), thrs=np.pi/2)
            dtheta_vel = np.append(0, np.diff(theta_vel))

            dthetasum_vel = np.convolve(dtheta_vel, kernel, 'same')

            peaks_vel = argrelextrema(abs(dthetasum_vel), np.greater)[0]
            turns_vel = peaks_vel[ (abs(dthetasum_vel[peaks_vel]) > thrs).nonzero()[0] ]

            plot_turnrates(events, dthetasum_shape, dthetasum_vel, turns_shape, turns_vel, pp, _title, fps=fps)

            _temp = np.zeros_like(dtheta_shape)
            _temp[turns_shape] = 1
            turns_shape_array = _temp

            _temp = np.zeros_like(dtheta_vel)
            _temp[turns_vel] = 1
            turns_vel_array = _temp
                                    
            # plot swim direction analysis
            fig = figure(figsize=(12,8), facecolor='w')
            ax1 = subplot(211)
            ax1.imshow(TVbg, cmap=cm.gray) # TVbg is clip out of ROI
            ax1.plot(x[rng]-TVx1, y[rng]-TVy1, 'gray')
            ax1.plot(water_x[t-preRange:t]-TVx1, water_y[t-preRange:t]-TVy1, 'c.')
            if matched:
                ax1.plot(   water_x[t:matched[0]]-TVx1, 
                            water_y[t:matched[0]]-TVy1, 'g.')
                ax1.plot(   water_x[matched[0]:matched[0]+preRange/4]-TVx1, 
                            water_y[matched[0]:matched[0]+preRange/4]-TVy1, 'r.')
            xlim([0, TVx2-TVx1]); ylim([TVy2-TVy1, 0])
            title(_title)
            
            ax2 = subplot(212)
            ax2.plot( swimdir_within )
            ax2.plot( peaks_within*1.15-0.1, 'mo' )
            if matched:
                xmin, xmax = t-preRange-10*fps, matched[0]+preRange/4
            else:
                xmin, xmax = t-preRange-10*fps, t+preRange/2+10*fps
            gzcs = np.cumsum(swimdir_within)
            gzcs -= gzcs[xmin]
            ax2.plot( gzcs/gzcs[xmax] )
            drawLines(0,1.2, events)
            ylim([0,1.2])
            xlim([xmin, xmax])
            ylabel('|: SwimDirection\no: approach events')

            data[fish][CS].append( {
                    'fname' : fname,
                    'x': x[rng], 'y': y[rng], 'z': z[rng], 
                    'X': X[rng], 'Y': Y[rng], 'Z': Z[rng], # calibrate space (mm)
                    'speed3D': _speed3D,                   # calibrate space (mm)
                    'movingSTD' : _movingSTD,              # calibrate space (mm)
                    'd2inflow': _d2inflow,                 # calibrate space (mm)
                    'ringpixels': _ringpixels,
                   
                    'peaks_within': peaks_within[rng],
                    'xy_within': xy_within[rng],
                    'location_one_third' : location_one_third[rng],
                    'swimdir_within' : swimdir_within[rng],
                    
                    'dtheta_shape': dtheta_shape[rng],
                    'dtheta_vel': dtheta_vel[rng],
                    'turns_shape': turns_shape_array[rng], # already +/- preRange
                    'turns_vel': turns_vel_array[rng],
                    
                    'events' : events,
                    'matchedUSname' : matchedUSname,
                    'TVroi' : (TVx1,TVy1,TVx2,TVy2),
                    'SVroi' : (SVx1,SVy1,SVx2,SVy2),
                    } )
            if pp:
                fig.savefig(pp, format='pdf')
            close('all') # release memory ASAP!
            
    if pp:
        pp.close()

def getPDFs(pickle_files, fishnames=None, createPDF=True):

    # type checking args
    if type(pickle_files) is str:
        pickle_files = [pickle_files]
    
    # convert to a list or set of fish names
    if type(fishnames) is str:
        fishnames = [fishnames]
    elif not fishnames:
        fishnames = set()

    # re-organize trials into a dict "data"
    data = {}
    # figure out trial number (sometime many trials in one files) for each fish
    # go through all pickle_files and use timestamps of file to sort events.
    timestamps = []
    
    for fp in pickle_files:
        # collect ctime of pickled files
        fname = os.path.basename(fp).split('.')[0] + '.avi'
        timestamps.append( time.strptime(fname, "%b-%d-%Y_%H_%M_%S.avi") )
        # look into the pickle and collect fish analyzed
        params = np.load(fp) # loading pickled file!
        if type(fishnames) is set:
            for fish in [fs for fl,fs in params.keys() if fl == fname and fs != 'mog']:
                fishnames.add(fish)
    timestamps = sorted(range(len(timestamps)), key=timestamps.__getitem__)

    # For each fish, go thru all pickled files
    for fish in fishnames:
        data[fish] = {}

        # now go thru the sorted
        for ind in timestamps:
            
            fp = pickle_files[ind]
            print 'processing #%d\n%s' % (ind, fp)
            add2DataAndPlot(fp, fish, data, createPDF)

    return data

def plotTrials(data, fish, CSname, key, step, offset=0, pp=None):
    fig = figure(figsize=(12,8), facecolor='w')
    
    ax1 = fig.add_subplot(121) # raw trace
    ax2 = fig.add_subplot(222) # learning curve
    ax3 = fig.add_subplot(224) # bar plot
    preP, postP, postP2 = [], [], []
    longestUS = 0
    for n, measurement in enumerate(data[fish][CSname]):
        tr = n+1
        CS, USs, preRange = measurement['events']
        subplot(ax1)
        mi = -step*(tr-1)
        ma = mi + step
        drawLines(mi, ma, (preRange, [preRange+(USs[0]-CS)], preRange))
        longestUS = max([us-CS+preRange*3/2 for us in USs]+[longestUS])
        
        # 'measurement[key]': vector around the CS timing (+/-) preRange. i.e., preRange is the center
        ax1.plot(measurement[key]-step*(tr-1)+offset)
        title(CSname+': '+key)                                                                  # cf. preRange = 3600 frames
        pre = measurement[key][:preRange].mean()+offset                                       # 2 min window
        post = measurement[key][preRange:preRange+(USs[0]-CS)].mean()+offset                  # 23 s window
        post2 = measurement[key][preRange+(USs[0]-CS):preRange*3/2+(USs[0]-CS)].mean()+offset # 1 min window after US
        preP.append(pre)
        postP.append(post)
        postP2.append(post2)

        ax3.plot([1, 2, 3], [pre, post, post2],'o-')

    ax1.set_xlim([0,longestUS])
    ax1.axis('off')

    subplot(ax2)
    x = range(1, tr+1)
    y = np.diff((preP,postP), axis=0).ravel()
    ax2.plot( x, y, 'ko-', linewidth=2 )
    ax2.plot( x, np.zeros_like(x), '-.', linewidth=1, color='gray' )
    # grid()
    slope, intercept, rvalue, pval, stderr = stats.stats.linregress(x,y) 
    title('slope = zero? p-value = %f' % pval)
    ax2.set_xlabel("Trial#")
    ax2.set_xlim([0.5,tr+0.5])
    ax2.set_ylabel('CS - pre')
    
    subplot(ax3)
    ax3.bar([0.6, 1.6, 2.6], [np.nanmean(preP), np.nanmean(postP), np.nanmean(postP2)], facecolor='none')
    t, pval = stats.ttest_rel(postP, preP)
    title('paired t p-value = %f' % pval)
    ax3.set_xticks([1,2,3])
    ax3.set_xticklabels(['pre', CSname, measurement['matchedUSname']])
    ax3.set_xlim([0.5,3.5])
    ax3.set_ylabel('Raw mean values')

    tight_layout(2, h_pad=1, w_pad=1)
    
    if pp:
        fig.savefig(pp, format='pdf')
    close('all')

    return np.vstack((preP, postP, postP2))

def getSummary(data, dirname=None):

    for fish in data.keys():
        for CSname in data[fish].keys():

            if dirname:
                pp = PdfPages(os.path.join(dirname, '%s_for_%s.pdf' % (CSname,fish)))
                print 'generating %s_for_%s.pdf' % (CSname,fish)

            book = Workbook()
            sheet1 = book.add_sheet('speed3D')
            avgs = plotTrials(data, fish, CSname, 'speed3D', 30, pp=pp)
            putNp2xls(avgs, sheet1)

            sheet2 = book.add_sheet('d2inflow')
            avgs = plotTrials(data, fish, CSname, 'd2inflow', 200, pp=pp)
            putNp2xls(avgs, sheet2)

            # sheet3 = book.add_sheet('smoothedz')
            sheet3 = book.add_sheet('Z')
            # avgs = plotTrials(data, fish, CSname, 'smoothedz', 100, pp=pp)
            avgs = plotTrials(data, fish, CSname, 'Z', 30, pp=pp)
            putNp2xls(avgs, sheet3)

            sheet4 = book.add_sheet('ringpixels')
            avgs = plotTrials(data, fish, CSname, 'ringpixels', 1200, pp=pp)
            putNp2xls(avgs, sheet4)
            
            sheet5 = book.add_sheet('peaks_within')
            avgs = plotTrials(data, fish, CSname, 'peaks_within', 1.5, pp=pp)
            putNp2xls(avgs, sheet5)
            
            sheet6 = book.add_sheet('swimdir_within')
            avgs = plotTrials(data, fish, CSname, 'swimdir_within', 1.5, pp=pp)
            putNp2xls(avgs, sheet6)
            
            sheet7 = book.add_sheet('xy_within')
            avgs = plotTrials(data, fish, CSname, 'xy_within', 1.5, pp=pp)
            putNp2xls(avgs, sheet7)
            
            sheet8 = book.add_sheet('turns_shape')
            avgs = plotTrials(data, fish, CSname, 'turns_shape', 1.5, pp=pp)
            putNp2xls(avgs, sheet8)
            
            sheet9 = book.add_sheet('turns_vel')
            avgs = plotTrials(data, fish, CSname, 'turns_vel', 1.5, pp=pp)
            putNp2xls(avgs, sheet9)
            
            if dirname:
                pp.close()
                book.save(os.path.join(dirname, '%s_for_%s.xls' % (CSname,fish)))
                close('all')
            else:
                show()




def add2Pickles(dirname, pickle_files):
    # dirname : folder to look for pickle files
    # pickle_files : output, a list to be concatenated.
    pattern = os.path.join(dirname, '*.pickle')
    temp = [_ for _ in glob(pattern) if not _.endswith('- Copy.pickle') and 
                    not os.path.basename(_).startswith('Summary')]
    pickle_files += temp
    

if __name__ == '__main__':
    
    pickle_files = []
    
    # small test data
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Aug4/test', pickle_files)
    # outputdir = 'R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Aug4/test'

    # NeuroD fish Aug4
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Aug4/2015-08-04', pickle_files)
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Aug4/2015-08-05', pickle_files)
    # outputdir = 'R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Aug4'
    
    # IN24 fish
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/IN24/2015-09-30', pickle_files)
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/IN24/2015-10-01', pickle_files)
    # outputdir = 'R:/Data/itoiori/behav/adult whitlock/conditioning/IN24'

    # IN25 2days
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/IN25 2days/2015-09-03', pickle_files)
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/IN25 2days/2015-09-04', pickle_files)
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/IN25 2days/2015-09-08', pickle_files)
    # outputdir = 'R:/Data/itoiori/behav/adult whitlock/conditioning/IN25 2days'

    # Cys Ala experiment summary
    # (1) IN10 2 females apetitive  CS+: L-ala 3x10-5M CS-: L-cys 3x10-5M
    # analyzed in old script, not by multitrack.py? no pickle found
    # (2) appetitive 2nd set        CS+: L-cys 3x10-5M CS-: L-ala 3x10-5M
    # analyzed in old script, not by multitrack.py? no pickle found
    # (3) NeuroD Nov3               CS+: L-cys 3x10-5M CS-: L-ala 3x10-5M
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Nov3/2015-11-03', pickle_files)
    # add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Nov3/2015-11-04', pickle_files)
    # outputdir = 'R:/Data/itoiori/behav/adult whitlock/conditioning/NeuroD/Nov3'
    # (4) Chie Nov05-12             CS+: L-cys 3x10-5M CS-: L-ala 3x10-5M
    add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/Chie/2015-11-10', pickle_files)
    add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/Chie/2015-11-11', pickle_files)
    add2Pickles('R:/Data/itoiori/behav/adult whitlock/conditioning/Chie/2015-11-12', pickle_files)
    outputdir = 'R:/Data/itoiori/behav/adult whitlock/conditioning/Chie'
    # N=8

    # show me what you got
    for pf in pickle_files:
        print pf
    
    fp = os.path.join(outputdir, 'Summary.pickle')
    createPDF = True # useful when plotting etc code updated
    if 1: # refresh analysis
        data = getPDFs(pickle_files, createPDF=createPDF)
        import cPickle as pickle
        with open(os.path.join(outputdir, 'Summary.pickle'), 'wb') as f:
            pickle.dump(data, f)
    else: # or reuse previous
        data = np.load(fp)
    
    getSummary(data, outputdir)
    pickle2mat(fp, data)

