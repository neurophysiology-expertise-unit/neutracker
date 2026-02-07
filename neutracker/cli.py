
import argparse
import sys
import os
import json
import numpy as np
from neutracker.tracker import MPTracker
from neutracker.io import TiffFileSequence, NorpixFile, AVIFileSequence, exportResultsToHDF5
from neutracker.utils import find_outliers
from neutracker.parutils import par_process_tiff
import glob
from natsort import natsorted
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Neutracker CLI: Headless Mouse Pupil Tracker')
    parser.add_argument('file', help='Input video file (.tif sequence, .seq, .avi) or directory/pattern')
    parser.add_argument('--params', help='Path to parameters JSON file (exported from GUI)', required=True)
    parser.add_argument('--output', '-o', help='Output HDF5 file path. If not provided, defaults to [folder_name]_params.hdf5 in the data directory.')
    parser.add_argument('--parallel', '-m', action='store_true', help='Run in parallel mode (multiprocessing)')
    
    args = parser.parse_args()
    
    # Input Processing
    targetpath = os.path.abspath(args.file)
    
    print(f"Processing: {targetpath}")
    
    # Load parameters
    try:
        with open(args.params, 'r') as f:
            params = json.load(f)
        if 'crApprox' in params and params['crApprox']:
            params['crApprox'] = np.array(params['crApprox'])
        
        # Ensure defaults are populated in the main process params dict
        # This is critical for exportResultsToHDF5 if running in parallel
        defaults = MPTracker.defaults.copy()
        for k, v in defaults.items():
            if k not in params:
                params[k] = v
                
    except Exception as e:
        print(f"Error loading parameters: {e}")
        sys.exit(1)

    # Determine Output Filename
    if args.output:
        output_file = args.output
    else:
        # Resolve data_dir strictly
        if os.path.isdir(targetpath):
            data_dir = targetpath
        else:
            # Assume it's a file
            data_dir = os.path.dirname(targetpath)
        
        # If targetpath is just a filename "foo.tif" in current dir, dirname is empty ''
        if not data_dir:
            data_dir = '.'
            
        folder_name = os.path.basename(os.path.abspath(data_dir))
        
        # Sanity check if folder_name is empty (e.g. root)
        if not folder_name:
             folder_name = "output"
             
        output_file = os.path.join(data_dir, f"{folder_name}_params.hdf5")

    print(f"Output:     {output_file}")

    # Initialize Image Stack
    ext = os.path.splitext(targetpath)[1]
    imgstack = None
    
    try:
        if ext in ['.tif', '.tiff']:
            imgstack = TiffFileSequence(targetpath)
        elif ext == '.seq':
            imgstack = NorpixFile(targetpath)
        elif ext == '.avi':
            imgstack = AVIFileSequence(targetpath)
        else:
             # Fallback/Glob
            if '*' in targetpath:
                files = natsorted(glob.glob(targetpath))
                if len(files) > 0:
                     if files[0].endswith('.tif') or files[0].endswith('.tiff'):
                          # TiffFileSequence handles globs internally if passed wildcard, but here we expanded it?
                          # Actually TiffFileSequence expects path w/ wildcard OR first file
                          # Let's pass the wildcard string directly if TiffFileSequence supports it
                          # Looking at TiffFileSequence code: it assumes dir of file.
                          # Let's try passing the targetpath string directly if it has *
                          imgstack = TiffFileSequence(targetpath)
                     else:
                          # Assume AVI for now for others? Or sequence
                          imgstack = AVIFileSequence(files)
            else:
                 # Directory?
                 if os.path.isdir(targetpath):
                      # Guess extension?
                      tifs = glob.glob(os.path.join(targetpath, '*.tif'))
                      if tifs:
                           imgstack = TiffFileSequence(os.path.join(targetpath, '*.tif'))
                      else:
                           print("No supported files found in directory.")
                           sys.exit(1)
                 else:
                      print("Unknown input type.")
                      sys.exit(1)
    except Exception as e:
        print(f"Failed to initialize image stack: {e}")
        sys.exit(1)

    if imgstack is None:
        print("Could not load image stack.")
        sys.exit(1)
        
    print(f"Found {imgstack.nFrames} frames.")
    params['number_frames'] = imgstack.nFrames

    # Tracking
    results = {
        'ellipsePix': np.full((imgstack.nFrames, 5), np.nan, dtype=np.float32),
        'pupilPix': np.full((imgstack.nFrames, 2), np.nan, dtype=np.float32),
        'crPix': np.full((imgstack.nFrames, 2), np.nan, dtype=np.float32),
         # Reference points from params
        'reference': [params.get('points', [0,0,0,0])[0], params.get('points', [0,0,0,0])[2]] if 'points' in params and len(params['points'])==4 else [],
        'image_shape': imgstack.get(0).shape
    }

    if args.parallel:
        print("Running in PARALLEL mode.")
        # Parallel Mode
        # We need the list of filenames for par_process_tiff
        if not hasattr(imgstack, 'filenames') or not imgstack.filenames:
             print("Error: Parallel mode requires a sequence of files (TiffFileSequence).")
             sys.exit(1)
        
        # par_process_tiff returns a list of results [res1, res2, ...]
        # Each res is a tuple: (cr_pos, pupil_pos, pupil_radius, pupil_ellipse_par)
        try:
             # Make sure to import the function
             # call it
             res_list = par_process_tiff(imgstack.filenames, params, verbose=True)
             
             # Parse results back into numpy arrays
             # parutils returns: return (outimg, cr_pos, pupil_pos, pupil_radius, pupil_ellipse_par) ?? 
             # Wait, let's check tracker.py return signature!
             # tracker.apply returns: return res[1:] -> (cr_pos, pupil_pos, pupil_radius, pupil_ellipse_par)
             # par_process_tiff calls tracker.apply(frame) and appends result
             
             # So results list structure is: [ (cr, pup, rad, ell), (cr, pup, rad, ell), ... ]
             
             results['ellipsePix'][:, :2] = np.array([r[2] for r in res_list]) # pupil_radius
             results['ellipsePix'][:, 2:] = np.array([r[3] for r in res_list]) # pupil_ellipse_par
             results['pupilPix'][:, :] = np.array([r[1] for r in res_list], dtype=np.float32)
             results['crPix'][:, :] = np.array([r[0] for r in res_list], dtype=np.float32)
             
        except Exception as e:
             print(f"Parallel processing failed: {e}")
             import traceback
             traceback.print_exc()
             sys.exit(1)
    else:
        print("Running in SEQUENTIAL mode.")
        tracker = MPTracker(parameters=params, drawProcessedFrame=False)
        for f in tqdm(range(imgstack.nFrames), desc="Tracking"):
            img = imgstack.get(f)
            if img is None: continue
            
            # tracker.apply returns (cr_pos, pupil_pos, pupil_radius, pupil_ellipse_par)
            cr_pos, pupil_pos, pupil_radius, pupil_ellipse_par = tracker.apply(img)
            
            results['ellipsePix'][f, :2] = pupil_radius
            results['ellipsePix'][f, 2:] = pupil_ellipse_par
            results['pupilPix'][f, :] = pupil_pos
            results['crPix'][f, :] = cr_pos

    # Outlier Removal
    nanidx = find_outliers(results['pupilPix'][:, 0])
    if len(nanidx) > 0:
        print(f"Removing {len(nanidx)} outliers.")
        results['ellipsePix'][nanidx, :] = np.nan
        results['pupilPix'][nanidx, :] = np.nan

    # Save
    exportResultsToHDF5(output_file, params, results)
    print(f"Done.")

if __name__ == '__main__':
    main()
