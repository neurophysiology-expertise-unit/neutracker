
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
    parser = argparse.ArgumentParser(description='Neutracker CLI: Headless Mouse Pupil Tracker & Video Utility')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # TRACK Command
    track_parser = subparsers.add_parser('track', help='Track pupil in video files')
    track_parser.add_argument('file', help='Input video file (.tif sequence, .seq, .avi) or directory/pattern')
    track_parser.add_argument('--params', help='Path to parameters JSON file (exported from GUI)', required=True)
    track_parser.add_argument('--output', '-o', help='Output HDF5 file path.')
    track_parser.add_argument('--parallel', '-m', action='store_true', help='Run in parallel mode')

    # CONVERT Command
    convert_parser = subparsers.add_parser('convert', help='Convert between AVI and TIFF sequence')
    convert_parser.add_argument('file', nargs='?', help='Input file or directory (supports batch processing of subfolders)')
    convert_parser.add_argument('--output', '-o', help='Output directory or file')
    convert_parser.add_argument('--contrast', type=int, default=255, help='Upper limit for contrast stretching (e.g., 60)')
    convert_parser.add_argument('--config', '-c', help='Path to JSON configuration file for batch conversion')

    # Backward compatibility: if first arg is not a command, assume 'track'
    if len(sys.argv) > 1 and sys.argv[1] not in ['track', 'convert', '-h', '--help']:
        sys.argv.insert(1, 'track')

    args = parser.parse_args()

    if args.command == 'convert':
        from neutracker.converter import avi_to_tiff_sequence, tiff_sequence_to_avi
        try:
            if args.config:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                
                tasks = config.get('conversions', [])
                print(f"Loaded {len(tasks)} conversion tasks from config.")
                
                for task in tasks:
                    target_pattern = task.get('input')
                    out_path = task.get('output')
                    contrast = task.get('contrast', args.contrast)
                    
                    if not target_pattern or not out_path:
                        print(f"Skipping invalid task (missing input/output): {task}")
                        continue
                        
                    matched_targets = natsorted(glob.glob(target_pattern))
                    
                    if not matched_targets:
                        print(f"Warning: No folders matched the pattern: {target_pattern}")
                        continue
                        
                    for target in matched_targets:
                        # Determine actual output path for this specific target
                        if len(matched_targets) > 1 or os.path.isdir(out_path) or not out_path.lower().endswith('.avi'):
                            # If it's a batch job, out_path must act as a directory
                            out_dir = os.path.dirname(out_path) if out_path.lower().endswith('.avi') else out_path
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir, exist_ok=True)
                                
                            folder_name = os.path.basename(target.rstrip('\\/'))
                            actual_out = os.path.join(out_dir, f"{folder_name}.avi")
                        else:
                            actual_out = out_path
                            
                        print(f"\n>>> Batch Processing from config: {os.path.basename(target)}")
                        try:
                            tiff_sequence_to_avi(target, actual_out, v_max=contrast)
                        except Exception as sub_e:
                            print(f"Error processing {target}: {sub_e}")
                return

            if not args.file or not args.output:
                print("Error: For manual conversion, you must provide both an input 'file' and an '--output'. Or use '--config'.")
                sys.exit(1)

            target = os.path.abspath(args.file)
            
            # 1. Handle AVI -> TIFF
            if target.lower().endswith('.avi'):
                avi_to_tiff_sequence(target, args.output)
                return

            # 2. Handle TIFF -> AVI (Single or Batch)
            if os.path.isdir(target):
                tifs = glob.glob(os.path.join(target, "*.tif*"))
                if tifs:
                    # Single folder processing
                    out_path = args.output
                    if not out_path.lower().endswith('.avi'):
                        if os.path.isdir(out_path):
                            out_path = os.path.join(out_path, f"{os.path.basename(target)}.avi")
                        else:
                            out_path += ".avi"
                    tiff_sequence_to_avi(target, out_path, v_max=args.contrast)
                else:
                    # BATCH MODE: Iterate through subdirectories
                    print(f"No TIFFs in parent folder. Starting BATCH mode for subfolders in: {target}")
                    subdirs = [os.path.join(target, d) for d in os.listdir(target) if os.path.isdir(os.path.join(target, d))]
                    
                    if not os.path.exists(args.output):
                        os.makedirs(args.output)
                        
                    for s in subdirs:
                        sub_tifs = glob.glob(os.path.join(s, "*.tif*"))
                        if not sub_tifs: continue
                        
                        folder_name = os.path.basename(s)
                        out_path = os.path.join(args.output, f"{folder_name}.avi")
                        print(f"\n>>> Batch Processing: {folder_name}")
                        try:
                            tiff_sequence_to_avi(s, out_path, v_max=args.contrast)
                        except Exception as sub_e:
                            print(f"Error processing {folder_name}: {sub_e}")
            else:
                # File pattern
                tiff_sequence_to_avi(target, args.output, v_max=args.contrast)
        except Exception as e:
            print(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Tracking Logic (existing)
    targetpath = os.path.abspath(args.file)
    print(f"Processing: {targetpath}")
    
    # Load parameters
    try:
        with open(args.params, 'r') as f:
            params = json.load(f)
        if 'crApprox' in params and params['crApprox']:
            params['crApprox'] = np.array(params['crApprox'])
        
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
        if os.path.isdir(targetpath):
            data_dir = targetpath
        else:
            data_dir = os.path.dirname(targetpath)
        
        if not data_dir:
            data_dir = '.'
            
        folder_name = os.path.basename(os.path.abspath(data_dir))
        if not folder_name:
             folder_name = "output"
             
        output_file = os.path.join(data_dir, f"{folder_name}_params.hdf5")

    print(f"Output:     {output_file}")

    # Initialize Image Stack
    ext = os.path.splitext(targetpath)[1].lower()
    imgstack = None
    
    try:
        if ext in ['.tif', '.tiff']:
            imgstack = TiffFileSequence(targetpath)
        elif ext == '.seq':
            imgstack = NorpixFile(targetpath)
        elif ext == '.avi':
            imgstack = AVIFileSequence(targetpath)
        else:
            if '*' in targetpath:
                imgstack = TiffFileSequence(targetpath)
            elif os.path.isdir(targetpath):
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

    # Results Initialization
    results = {
        'ellipsePix': np.full((imgstack.nFrames, 5), np.nan, dtype=np.float32),
        'pupilPix': np.full((imgstack.nFrames, 2), np.nan, dtype=np.float32),
        'crPix': np.full((imgstack.nFrames, 2), np.nan, dtype=np.float32),
        'reference': [params.get('points', [0,0,0,0])[0], params.get('points', [0,0,0,0])[2]] if 'points' in params and len(params['points'])==4 else [],
        'image_shape': imgstack.get(0).shape
    }

    if args.parallel:
        print("Running in PARALLEL mode.")
        if not hasattr(imgstack, 'filenames') or not imgstack.filenames:
             print("Error: Parallel mode requires a sequence of files (TiffFileSequence).")
             sys.exit(1)
        
        try:
             res_list = par_process_tiff(imgstack.filenames, params, verbose=True)
             results['ellipsePix'][:, :2] = np.array([r[2] for r in res_list])
             results['ellipsePix'][:, 2:] = np.array([r[3] for r in res_list])
             results['pupilPix'][:, :] = np.array([r[1] for r in res_list], dtype=np.float32)
             results['crPix'][:, :] = np.array([r[0] for r in res_list], dtype=np.float32)
        except Exception as e:
             print(f"Parallel processing failed: {e}")
             sys.exit(1)
    else:
        print("Running in SEQUENTIAL mode.")
        tracker = MPTracker(parameters=params, drawProcessedFrame=False)
        for f in tqdm(range(imgstack.nFrames), desc="Tracking"):
            img = imgstack.get(f)
            if img is None: continue
            cr_pos, pupil_pos, pupil_radius, pupil_ellipse_par = tracker.apply(img)
            results['ellipsePix'][f, :2] = pupil_radius
            results['ellipsePix'][f, 2:] = pupil_ellipse_par
            results['pupilPix'][f, :] = pupil_pos
            results['crPix'][f, :] = cr_pos

    # Save
    exportResultsToHDF5(output_file, params, results)
    print(f"Done.")

if __name__ == '__main__':
    main()
