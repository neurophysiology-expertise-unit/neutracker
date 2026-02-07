#! /usr/bin python
# IO utilities for mptracker.
# Aim is to know the number of frames in advance and have a common interface for asking image frames
# Supported formats:
#    - multipage TIFF
#    - streamPIX seq files
#    - avi files (on demmand)
# November 2016 - Joao Couto

import sys
import os
from os.path import join as pjoin
import numpy as np
from glob import glob
from tifffile import TiffFile,imread
from tempfile import mkdtemp
from shutil import copyfile
import cv2 
import re
import json
from .utils import *

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

def createResultsFile(filename,nframes,npoints = 4,MPIO = False):
    if len(os.path.dirname(filename)) > 0 and not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    import h5py as h5
    if MPIO:
        f = h5.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    else:
        f = h5.File(filename, 'w')
    f.create_dataset('diameter',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('azimuth',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('elevation',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('theta',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('ellipsePix',dtype = np.float32,
                     shape=(nframes,5),compression = 'gzip')
    f.create_dataset('positionPix',dtype = np.float32,
                     shape=(nframes,2),compression = 'gzip')
    f.create_dataset('crPix',dtype = np.float32,
                     shape=(nframes,2),compression = 'gzip')
    f.create_dataset('pointsPix',dtype = np.float32,
                     shape=(npoints,2),compression = 'gzip')
    return f

def exportResultsToHDF5(resultfile,
                        parameters,
                        results):
    '''
    TODO: Make this simpler...
    '''
    fname,ext = os.path.splitext(resultfile)
    if len(ext)==0:
        resultfile += '.hdf5'
    if os.path.isfile(resultfile):
        print('Overwriting file: {0}'.format(resultfile))
    diam = computePupilDiameterFromEllipse(
        results['ellipsePix'],
        computeConversionFactor(results['reference'],
                                2.*parameters['eye_radius_mm']))
    if parameters.get('fov_mm', 0) > 0:
        # If FOV is provided in mm (e.g. 300mm for 30cm arena)
        # Calculate conversion factor as mm_per_pixel
        # image_shape is (height, width)
        if 'image_shape' in results:
            img_width = results['image_shape'][1]
            cFactor = parameters['fov_mm'] / float(img_width)
            print(f'Using FOV {parameters["fov_mm"]}mm -> {cFactor:.4f} mm/px')
            # Recalculate diameter with new factor
            diam = computePupilDiameterFromEllipse(
                results['ellipsePix'],
                cFactor)
        else:
             print("Warning: FOV set but image_shape not found in results. Using default calibration.")
             cFactor = computeConversionFactor(results['reference'],
                                    2.*parameters['eye_radius_mm'])

    if parameters['crTrack']:
        az,el,theta = convertPixelToEyeCoords(results['pupilPix'],
                                              results['reference'],
                                              results['crPix'],
                                              eyeDiameterEstimate =
                                              2*parameters['eye_radius_mm'])
    else:
        az,el,theta = convertPixelToEyeCoords(results['pupilPix'],
                                              results['reference'],
                                              eyeDiameterEstimate =
                                              2*parameters['eye_radius_mm'])
    fd = createResultsFile(resultfile,
                           len(diam),
                           npoints = len(parameters['points']))
    fd['ellipsePix'][:] = results['ellipsePix'].astype(np.float32)
    fd['positionPix'][:] = results['pupilPix'].astype(np.float32)
    fd['crPix'][:] = results['crPix'].astype(np.float32)
    fd['diameter'][:] = diam.astype(np.float32)
    fd['azimuth'][:] = az.astype(np.float32)
    fd['elevation'][:] = el.astype(np.float32)
    fd['theta'][:] = theta.astype(np.float32)
    
    # Save position_mm if FOV is provided
    if parameters.get('fov_mm', 0) > 0 and 'image_shape' in results:
        img_width = results['image_shape'][1]
        cFactor = parameters['fov_mm'] / float(img_width)
        # positionPix is (y, x) or (x, y)? 
        # tracker.py: pupil_pos = np.array([ellipse[0][0],ellipse[0][1]]) -> (x, y)
        # so we just multiply by cFactor
        pos_mm = results['pupilPix'] * cFactor
        if 'position_mm' not in fd:
             fd.create_dataset('position_mm', dtype=np.float32, shape=(len(diam), 2), compression='gzip')
        fd['position_mm'][:] = pos_mm.astype(np.float32)

    fd['pointsPix'][:] = np.array(parameters['points'])
    fd.flush()
    fd.close()
    saveTrackerParameters(resultfile,parameters)
    return True

def saveTrackerParameters(paramfile,parameters):
    fname,ext = os.path.splitext(paramfile)
    if fname is None:
        fname =  paramfile
    if len(fname)==0:
        print('Can not save to file with no name...'+paramfile + ' Crack...')
        return False
    import json
    from .io import JsonEncoder
    paramfile = fname + '.json'
    with open(paramfile,'w') as f:
        tmp = dict(parameters)
        json.dump(tmp,f,indent=4, sort_keys=True,cls=JsonEncoder)
    return True


def copyFilesToTmp(targetpath):
    path = os.path.dirname(targetpath)
    basename, extension = os.path.splitext(os.path.basename(targetpath))
    for f in range(len(basename)):
        if not basename[-f].isdigit():
            break
    if not -f+1 == 0:
        f = -f+1
    else:
        f = -1
    basename = basename[:f]
    fnames = np.sort(glob(pjoin(path,'*' + extension)))
    filenames = []
    for f in fnames:
        if basename in f:
            filenames.append(f)
    if not len(filenames):
        print('Wrong target path: ' + path + '*' + extension)
        raise
    tempdir = mkdtemp(basename)
    print('Using temporary folder: ' + tempdir)
    for f in filenames:
        target_f = os.path.basename(f)
        copyfile(f,pjoin(tempdir,target_f))
        print('Copied '+target_f)
    return pjoin(tempdir,os.path.basename(filenames[0]))

class TiffFileSequence(object):
    def __init__(self,targetpath = None):
        '''Lets you access a sequence of TIFF files without noticing...'''
        from natsort import natsorted
        if '*' in targetpath:
            filenames = natsorted(glob(targetpath))
            self.filenames = filenames
            self.path = os.path.dirname(self.filenames[0])
            self.basename,extension = os.path.splitext(os.path.basename(self.filenames[0]))
        else:
            self.path = os.path.dirname(targetpath)
            print(self.path)
            self.basename,extension = os.path.splitext(os.path.basename(targetpath))
            self.filenames = natsorted(glob(pjoin(self.path,'*'+extension)))
            print(len(self.filenames))
        if not len(self.filenames):
            print('Wrong target path: ' + pjoin(self.path,'*' + extension))
            raise
        self.files = []
        framesPerFile = []
        for i,f in enumerate(self.filenames):
            if i==0 or i== len(self.filenames)-1:
                self.files.append(TiffFile(f))
                self.compressedhack = False
                try:
                    N,h,w = self.files[i].series[0].shape
                except:
                    h,w = self.files[i].series[0].shape
                    N = len(self.files[i].series)
                    if N > 1:
                        self.compressedhack = True
            else:
                self.files.append(None)
            framesPerFile.append(np.int64(N))
            if 'h' in dir(self):
                if not self.h == h:
                    print('Wrong height value on one of the files.')
                    raise
            else:
                self.h = h
                self.w = w
        self.framesPerFile = np.array(framesPerFile, dtype=np.int64)
        self.framesOffset = np.hstack([0,np.cumsum(self.framesPerFile[:-1])])
        self.nFrames = np.sum(framesPerFile)
        print('There are {0} frames in {1} files'.format(self.nFrames,
                                                         len(self.filenames)))
        self.curimg = None
        self.curidx = -1
    def getFrameIndex(self,frame):
        '''Computes the frame index from multipage tiff files.'''
        fileidx = np.where(self.framesOffset <= frame)[0][-1]
        # This breaks for huge tif files
        return fileidx,int(frame - self.framesOffset[fileidx])

    def getDescripion(self,frame):
        '''Gets image description tag from tiff page'''
        frameidx = self.getFrameIndex(frame)
        tags = self.files[fileidx][frameidx].tags
        if 'image_description' in tags.keys():
            return tags['image_description']
        else:
            return None

    def get(self,frame):
        '''Returns an image given the frame ID.
        Useful attributes are nFrames, h (frame height) and w (frame width)
        '''
        fileidx,frameidx = self.getFrameIndex(frame)
        if len(self.filenames) > 10: # do memmap
            if self.files[fileidx] is None:
                self.files[fileidx] = TiffFile(self.filenames[fileidx])
            if not self.files[fileidx-1] is None:
                self.files[fileidx-1].close()
                self.files[fileidx-1] = None
            if self.compressedhack:
                img = self.files[fileidx].series[frameidx].asarray()
            else:
                img = self.files[fileidx].asarray(frameidx)
        else:
            if not self.curidx == fileidx:
                self.curimg = imread(self.filenames[fileidx])
                self.curidx = fileidx
            img = self.curimg[frameidx]
        if img.dtype == np.uint16:
            img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        return img

    def close(self):
        for fd in self.files:
            fd.close()

class NorpixFile(object):
    def __init__(self,targetpath = None,extension='tif'):
        '''Wrapper to norpix seq files'''
        self.path = os.path.dirname(targetpath)
        self.filenames = [targetpath]
        from pims import NorpixSeq
        self.files = [NorpixSeq(f) for f in self.filenames]
        framesPerFile = []
        for f in self.files:
            N = len(f)
            framesPerFile.append(np.int64(N))
            if not 'h' in dir(self):
                h,w = (f.height, f.width)
                self.h = h
                self.w = w
        self.framesPerFile = np.array(framesPerFile, dtype=np.int64)
        self.framesOffset = np.hstack([0,np.cumsum(self.framesPerFile[:-1])])
        self.nFrames = sum(framesPerFile)

    def getFrameIndex(self,frame):
        '''Computes the frame index from multipage tiff files.'''
        fileidx = np.where(self.framesOffset <= frame)[0][-1]
        return fileidx,frame - self.framesOffset[fileidx]

    def get(self,frame):
        '''Returns an image given the frame ID.
        Useful attributes are nFrames, h (frame height) and w (frame width)
        '''
        fileidx,frameidx = self.getFrameIndex(frame)
        return self.files[fileidx].get_frame(frameidx)

    def close(self):
        for fd in self.files:
            fd.close()

# AVI
class AVIFileSequence(object):
    def __init__(self,targetpath = None):
        '''Lets you access a sequence of AVI files without noticing...'''
        if type(targetpath) is list:
            self.path = os.path.dirname(targetpath[0])
            self.filenames = targetpath
        else:
            self.path = os.path.dirname(targetpath)
            self.filenames = [targetpath]
        #self.basename,extension = os.path.splitext(os.path.basename(targetpath))
        #filenames = np.sort(glob(pjoin(self.path,'*'+extension)))
        # Use natural sort
        #pat = re.compile('([0-9]+)')
        #self.filenames = [f for f in np.sort(filenames)]
#        self.filenames = [filenames[j] for j in np.lexsort(np.array([[int(i) for i in pat.findall(os.path.basename(fname))] for fname in filenames]).T)]

        #if not len(self.filenames):
        #    print('Wrong target path: ' + pjoin(self.path,'*' + extension))
        #    raise
        self.files = []
        framesPerFile = []
        self.curidx = []
        for i,f in enumerate(self.filenames):
            self.files.append(VideoReader(f))
            N =  int(len(self.files[-1]))
            im = self.files[-1][0].asnumpy()
            h = int(im.shape[1])
            w = int(im.shape[0])
            framesPerFile.append(np.int64(N))
            self.curidx.append(0)
            if 'h' in dir(self):
                if not self.h == h:
                    print('Wrong height value on one of the files.')
                    raise
            else:
                self.h = h
                self.w = w
        self.framesPerFile = np.array(framesPerFile, dtype=np.int64)
        self.framesOffset = np.hstack([0,np.cumsum(self.framesPerFile[:-1])])
        self.nFrames = np.sum(framesPerFile)

    def getFrameIndex(self,frame):
        '''Computes the frame index from multiple files.'''
        fileidx = np.where(self.framesOffset <= frame)[0][-1]
        return fileidx,int(frame - self.framesOffset[fileidx])

    def get(self,frame):
        '''Returns an image given the frame ID.
        Useful attributes are nFrames, h (frame height) and w (frame width)
        '''
        fileidx,frameidx = self.getFrameIndex(frame)
        #if not self.curidx[fileidx] == frameidx:
        #    self.files[fileidx].set(1,frameidx)
        self.curidx[fileidx] = frameidx
        self.curidx[fileidx] +=1
        
        img = self.files[fileidx][frameidx].asnumpy()
        if len(img.shape) == 3:
            img = img[:,:,0]
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint16:
            img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        return img

    def close(self):
        for fd in self.files:
            del fd
