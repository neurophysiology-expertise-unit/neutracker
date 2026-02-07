import pylab as plt
import numpy as np
from .utils import *
from .tracker import *

def plot_results(results,parameters,img = None,ii = 100):
    fig = plt.figure(figsize = [10,3])
    if not img is None:
        import cv2
        clahe = cv2.createCLAHE(7,(10,10))
        img = clahe.apply(img)
        ax = fig.add_axes([0.025,0.05,0.25,0.95],aspect='equal')
        ax.imshow(img,cmap='gray',aspect='equal')
    if not 'reference' in results.keys():
        results['reference'] = [parameters['points'][0],
                                parameters['points'][2]]
    eyeCorners  = results['reference']
    reference = [eyeCorners[0][0] +
                 np.diff([eyeCorners[0][0],eyeCorners[1][0]])/2.,
                 eyeCorners[0][1] +
                 np.diff([eyeCorners[0][1],eyeCorners[1][1]])/2.]
    ax.plot(reference[0],reference[1],'g+',alpha=0.8,markersize=10,lw=1)
    ax.plot([results['reference'][0][0],results['reference'][1][0]],
            [results['reference'][0][1],results['reference'][1][1]],'-|y',
            alpha=0.8,markersize=25,lw=1)
    ax.plot(results['pupilPix'][ii,0],
            results['pupilPix'][ii,1],'r.',alpha=0.8)
    ax.plot(results['crPix'][ii,0],
            results['crPix'][ii,1],'bo',alpha=0.8)
    s1 = ellipseToContour(results['pupilPix'][ii,:],
                          results['ellipsePix'][ii,2]/2,
                          results['ellipsePix'][ii,3]/2,
                          results['ellipsePix'][ii,4],
                          np.linspace(0,2*np.pi,200))

    ax.plot(np.hstack([s1[:,0,1],s1[0,0,1]]),
            np.hstack([s1[:,0,0],s1[0,0,0]]),'-',color='orange',alpha=0.8)
    ax.grid(False)
    ax.set_axis_off();
    ax.axis('tight');
    
    axel = fig.add_axes([0.36,0.16,0.6,0.2])
    axdiam = fig.add_axes([0.36,0.76,0.6,0.2],sharex=axel)
    axaz = fig.add_axes([0.36,0.46,0.6,0.2],sharex=axel)
    
    diam = computePupilDiameterFromEllipse(
        results['ellipsePix']/2.,
        computeConversionFactor(results['reference'],
                                2*parameters['eye_radius_mm']))
    if parameters['crTrack']:
        az,el,theta = convertPixelToEyeCoords(
            results['pupilPix'],
            results['reference'],
            results['crPix'],
            eyeDiameterEstimate =2*parameters['eye_radius_mm'])
    else:
        az,el,theta = convertPixelToEyeCoords(
            results['pupilPix'],
            results['reference'],
            eyeDiameterEstimate =2*parameters['eye_radius_mm'])

    axdiam.plot(medfilt(diam));axdiam.set_xticklabels([])
    axaz.plot(medfilt(az));axaz.set_xticklabels([]);
    
    axaz.set_ylabel('Azimuth \n [deg]',color='black')
    axel.plot(medfilt(el));axel.set_ylabel('Elevation \n [deg]',color='black')
    
    axdiam.set_ylabel('Diameter \n [mm]',color='black')
    for a in [axdiam,axaz,axel]:
        cleanAx(a)
        a.axis('tight')
    #axel.set_ylim(np.array([-2.5,2.5])*np.nanstd(el) +
    #              np.nanmedian(el))
    #axaz.set_ylim(np.array([-2.5,2.5])*np.nanstd(az) +
    #              np.nanmedian(az))
    #axdiam.set_ylim(np.array([-2.5,2.5])*np.nanstd(diam) +
    #                np.nanmedian(diam))
    # axdiam.set_ylim([0,2.])
    # axaz.set_ylim([0,3.7])
    axel.set_xlabel('Frame number',color='black')
    plt.show()

def cleanAx(ax1):
    ax1.locator_params(axis='y',nbins=3)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.tick_params(axis='both', colors='black')

