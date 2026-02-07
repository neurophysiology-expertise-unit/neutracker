Neutracker
==========

**Neutracker** is a modernized version of the [mptracker](https://github.com/jpcouto/mptracker) repository, originally by Joao Couto. It has been updated to include:
- **Windows Executable**: Standalone `.exe` for easy deployment.
- **Headless CLI**: Command-line interface for batch processing and HPC environments.
- **Containerization**: Docker and Singularity/Apptainer support.
- **Modern Packaging**: `pyproject.toml` and `environment.yml` for reproducible builds.

Help making it work for your dataset by letting me know cases where it fails/works.

![picture](images/mptrackerExample.png)

Supported file formats:
-----------------------
   - Multipage TIFF sequence
   - Norpix seq files
   - AVI

Output file format:
-------------------

The output is an HDF5 file that can be read pretty much anywhere.

The file is organized as follows:

- **/diameter** - diameter of the pupil [estimated mm]
- **/azimuth** - azimuth of the pupil [estimated degrees]
- **/elevation** - elevation of the pupil [estimated degrees]
- */theta* - angle 
- */ellipsePix* - ellipse parameters for each frame [in pixels] [short_axis,long_axis,a,b,phi]
- */positionPix* - position of the eye in pixels (X, Y)
- **/position_mm** - position of the eye in mm (X, Y) [Only if FOV is provided]
- */crPix* - position of the corneal reflexion in pixels
- *points* - points marked by the user (left eye corner, top of the eye, right eye corner, bottom of the eye). These points mark the area to be analysed and define the scale.

In MATLAB do for example: `diam = h5read('filename.something','/diameter')`

Usage (GUI):
------------

**NOTE**: ``neutracker-gui --help`` for options.


Launch the GUI from the command line: ``neutracker-gui <filename>``. The filename is that of a seq file or one of the TIFF files in a TIFF sequence.  
### Instructions:

1.   Select left corner; top; right corner and bottom of the eye by dragging the points (keep the arangement between points the same as default).
2.   Once the points are in place click 'Update ROI'
3.   Adjust the parameters for best pupil contrast.
4.   Press 'Run' button or *r* key to launch the analysis. You will be prompted for a filename where to save  when it finishes. ( *r* stops the analysis also).
5.   Press the key *p* to plot results.

For other key options press the *h* key

### Command line options:

- *-o* <output file path> File where to save the results to (will ask if not specified).
- *-p* <parameter file> Parameter file to load.
- *--parallel* Run in parallel (CLI only).

### Apptainer / Singularity:

A definition file `neutracker.def` is provided for building a Singularity container.

```bash
# Build
apptainer build neutracker.sif neutracker.def

# Run
apptainer run neutracker.sif input.avi --params params.json
```


Installation:
-------------
Dependencies:

- PyQt5
- pyqtgraph
- opencv (cv2)
- h5py
- PIMS (for reading norpix seq files)
- tifffile
- natsort
- numpy, scipy, matplotlib
- imageio, slicerator, pillow

### Install instructions:

1. Get [ miniconda ](https://conda.io/miniconda.html)
2. Create environment: ``conda env create -f environment.yml``
3. Activate: ``conda activate neutracker``
4. Install: ``pip install .``

### Micromamba (Faster Alternative):

1. Create environment: ``micromamba create -f environment.yml``
2. Activate: ``micromamba activate neutracker``
3. Install: ``pip install .``

### Testing / Generating Data:

To verify the installation or the container build, you can generate synthetic data and run a test analysis.

```bash
# 1. Generate synthetic data (test_synthetic.avi)
# Using local python:
python3 tests/generate_synthetic_data.py test_synthetic.avi
# OR using Apptainer:
apptainer exec neutracker.sif python3 tests/generate_synthetic_data.py test_synthetic.avi

# 2. Run the analysis
# Using local install:
neutracker-gui test_synthetic.avi
# OR using Apptainer (headless):
apptainer run neutracker.sif test_synthetic.avi --params tests/test_params.json
```

**Please let me know whether this works for you and acknowledge if you use it in a publication.**

**Cagatay Aydin** - *cagjony@gmail.com*

February 2026
