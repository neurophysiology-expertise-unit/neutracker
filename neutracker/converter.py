import os
import cv2
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
import logging

def tiff_sequence_to_avi(input_pattern, output_path, fps=30, v_max=255, logger=None):
    """
    Stitches a sequence of TIFF files (including multi-page stacks) into an AVI video.
    Allows for contrast stretching by clipping at v_max and normalizing to 0-255.
    """
    from neutracker.io import TiffFileSequence
    import numpy as np
    
    # Auto-handle directory inputs
    if os.path.isdir(input_pattern):
        # Check if there are tiffs in this folder
        from glob import glob
        tifs = glob(os.path.join(input_pattern, "*.tif*"))
        if not tifs:
            # Maybe it's a parent folder? We'll handle this in the CLI, 
            # but for safety, return error here if no files.
            raise FileNotFoundError(f"No TIFF files found in: {input_pattern}")
        input_pattern = os.path.join(input_pattern, "*.tif*")
    
    try:
        imgstack = TiffFileSequence(input_pattern)
    except Exception as e:
        raise RuntimeError(f"Could not load TIFF sequence: {e}")

    if imgstack.nFrames == 0:
        raise FileNotFoundError(f"No frames found in: {input_pattern}")

    # Check if target already exists with same number of frames
    if os.path.exists(output_path):
        try:
            cap = cv2.VideoCapture(output_path)
            existing_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if existing_frames == imgstack.nFrames:
                msg = f"Target video '{output_path}' already exists with matching frame count ({existing_frames}). Skipping conversion."
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                imgstack.close()
                return
            else:
                msg = f"Target video exists but frame count ({existing_frames}) does not match source ({imgstack.nFrames}). Overwriting."
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
        except Exception as e:
            msg = f"Target video exists but could not verify frame count ({e}). Overwriting."
            if logger:
                logger.warning(msg)
            else:
                print(msg)

    if logger:
        logger.info(f"Found {imgstack.nFrames} total frames. Contrast: [0, {v_max}] -> [0, 255]")
    else:
        print(f"Found {imgstack.nFrames} total frames. Contrast: [0, {v_max}] -> [0, 255]")

    # Initialize VideoWriter using the first frame's dimensions
    first_frame = imgstack.get(0)
    h, w = first_frame.shape[:2]
    
    # Use MJPG for maximum compatibility with ImageJ
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    is_color = (len(first_frame.shape) == 3)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=is_color)

    for i in tqdm(range(imgstack.nFrames), desc="Stitching"):
        img = imgstack.get(i)
        if img is None:
            continue
        
        # Contrast Stretching
        if v_max < 255:
            img = np.clip(img, 0, v_max)
            img = (img.astype(np.float32) / v_max * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            # Standard normalization for 16-bit or other types if no v_max provided
            img = cv2.convertScaleAbs(img, alpha=(255.0/img.max()) if img.max() > 0 else 1.0)
            
        # VideoWriter expects BGR for color
        if is_color and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        out.write(img)
    
    out.release()
    imgstack.close()

def avi_to_tiff_sequence(input_path, output_dir, logger=None):
    """
    Extracts frames from an AVI video and saves them as a sequence of TIFF files.
    """
    if logger:
        logger.info(f"Opening video: {input_path}")
    else:
        print(f"Opening video: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize VideoReader
    try:
        vr = VideoReader(input_path, ctx=cpu(0))
        n_frames = len(vr)
        for i in tqdm(range(n_frames), desc="Extracting"):
            frame = vr[i].asnumpy()
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:06d}.tif"), frame)
    except Exception as e:
        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(n_frames), desc="Extracting"):
            ret, frame = cap.read()
            if not ret: break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:06d}.tif"), frame)
        cap.release()
