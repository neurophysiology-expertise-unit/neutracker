#! /usr/bin python
# Utilities for mptracker.
# Some of these are adapted from the Cox Lab EyeTracker.
# (https://github.com/coxlab/eyetracker)
# November 2016 - Joao Couto
from decord import VideoReader
import sys
import os
import numpy as np
import scipy.signal as signal
import cv2
from scipy.interpolate import interp1d

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def computePupilDiameterFromEllipse(ellipsePix,
                                    conversionFactor = None,
                                    smoothing = None):
    ''' diam = computePupilDiameterFromEllipse(ellipsePix,conversionFactor = None, smoothing = 'medfilt')
        ellipsePix is a Nx2 array (short_axis,long_axis)
        Compute the pupil diameter as the diameter of a circle with the same area as the fitted ellipse.
            Conversion factor (mm per pixel)
            Smoothing can be None, medfilt or sgolay.
    '''
    diam = np.sqrt(ellipsePix[:,0]*ellipsePix[:,1])*2
    if not conversionFactor is None:
        diam *= conversionFactor
    if smoothing is None or smoothing.lower() == 'none':
        return diam
    elif smoothing.lower() == 'medfilt':
        return medfilt(diam)
    elif smoothing.lower() == 'sgolay':
        from scipy.signal import savgol_filter
        return savgol_filter(diam, window_length = 5,
                             polyorder = 1, mode='nearest')
    
def computeConversionFactor(ref,estimate = 6.0):
    if ref is None or len(ref) < 2:
        return 1.0
    return float(estimate)/np.sqrt(np.diff([ref[0][0],ref[1][0]])**2. + np.diff([ref[0][1],ref[1][1]])**2.)

def convertPixelToEyeCoords(pupilPix,
                            eyeCorners,
                            crPix = None,
                            eyeDiameterEstimate = 3.0):
    if eyeCorners is None or len(eyeCorners) < 2:
        return np.full_like(pupilPix[:,0], np.nan), np.full_like(pupilPix[:,0], np.nan), np.full_like(pupilPix[:,0], np.nan)

    reference = [eyeCorners[0][0] +
                 np.diff([eyeCorners[0][0],eyeCorners[1][0]])/2.,
                 eyeCorners[0][1] +
                 np.diff([eyeCorners[0][1],eyeCorners[1][1]])/2.]
    pPix = np.zeros_like(pupilPix,dtype=np.float32)
    pPix[:] = np.nan 
    # Correct for movement of the entire eye.
    if not crPix is None:
        pPix[:,0] = pupilPix[:,0] - (crPix[:,0] - crPix[~np.isnan(crPix[:,0]),0][0])
        pPix[:,1] = pupilPix[:,1] - (crPix[:,1] - crPix[~np.isnan(crPix[:,1]),1][0])
    else:
        pPix[:,0] = pupilPix[:,0]
        pPix[:,1] = pupilPix[:,1]

    cFactor = computeConversionFactor(eyeCorners,eyeDiameterEstimate)
    [az,el,theta] = cart2sph((pPix[:,0]-reference[0])*cFactor,
                             (pPix[:,1]-reference[1])*cFactor,
                             eyeDiameterEstimate/2.)
    return np.rad2deg(az),np.rad2deg(el),np.rad2deg(theta)


def getEllipseMask(shape,ePos,eAxes,eAngles,dtype='uint8'):
    tmp = np.zeros(shape = shape,dtype=dtype)
    return cv2.ellipse(tmp,(int(ePos[0]),int(ePos[1])),(int(eAxes[0]),int(eAxes[1])),
                       eAngles[0],0,360,
                       255,-1)

def medfilt(x, k = 5):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    Borrowed from the web.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.nanmedian(y, axis=1)

def find_outliers(x,errthresh = 0.05,medfiltlen = 101):
    '''
    Find outliers by comparing to a median filtered version of the signal x
    '''
    assert medfiltlen % 2 == 1
    idx = np.where(~np.isnan(x))[0]
    from scipy.signal import medfilt
    med = medfilt(x[idx],medfiltlen)
    err = np.abs(x[idx]-med)/med
    return idx[err>errthresh]

def interp_nans(x):
    t = np.arange(len(x))
    return interp1d(t[~np.isnan(x)],x[~np.isnan(x)],
                    bounds_error = False,
                    fill_value='extrapolate')(t)

def sobel3x3(img,ksize = 5):
    """
    Compute vertical and horizontal gradients as well as the magnitude of image
    using openCV.
    Joao Couto - January 2017
    """
    gradx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=ksize)
    grady = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=ksize)
    mag = np.sqrt(np.power(gradx,2) + np.power(grady,2)) + 1e-16
    return mag,gradx,grady

#def pol2cart(theta, rho):
#    x = rho * np.cos(theta)
#    y = rho * np.sin(theta)
#    return x, y

#def cart2pol(x, y):
#    theta = np.arctan2(y, x)
#    rho = np.hypot(x, y)
#    return theta, rho


#############################################################################
######################FOR THE STARBURST ALGORITHM############################
#############################################################################
#try:
#    from scipy.weave import inline
#except:
#    try:
#        from weave import inline
#    except:
#        pass

def radial_transform(image, radii = None, alpha = 10, 
                     membuff=None):
    ''' Find points of interest in an image using the fast radial transform.
    (not that fast...)
    '''
    if not membuff is None and len(membuff) == 4:
        M,O,F,S = membuff
    else:
        M = np.zeros_like(image)
        O = np.zeros_like(image)
        F = np.zeros_like(image)
        S = np.zeros_like(image)
        
    (nrows, ncols) = image.shape
    if radii is None:
        radii = np.linspace(nrows/2000.,nrows/8.,6)
    mag,imgx,imgy = sobel3x3(image)
    imgx = imgx / mag
    imgy = imgy / mag
    (y, x) = np.mgrid[0:nrows, 0:ncols]   # grid
    ormagcode = """
Py_BEGIN_ALLOW_THREADS
                    
#define __TYPE  %s

int rows = Nmag[0];
//int rstart = 0;
//int rend = rows;
int tile_size = rows / n_tiles;
int rstart = (tile) * tile_size;
int rend;
if(tile == n_tiles-1){
    rend = rows;
} else {
    rend = (tile+1) * tile_size - 1;
}
int cols = Nmag[1];
int cstart = 0;
int cend = cols;
for(int r = rstart; r < rend; r++){
    for(int c = cstart; c < cend; c++){
        int index = r*cols + c;
        int posx_ = round(posx[index]);
        int posy_ = round(posy[index]);
        int negx_ = round(negx[index]);
        int negy_ = round(negy[index]);

        if(posx_ < 0 || posx_ > cols-1 ||
           posy_ < 0 || posy_ > rows-1 ||
           negx_ < 0 || negx_ > cols-1 ||
           negy_ < 0 || negy_ > rows-1){
            continue;
        }
        if(posx_ < 0) posx_ = 0;
        if(posx_ > cols-1) posx_ = cols-1;
        if(posy_ < 0) posy_ = 0;
        if(posy_ > rows-1) posy_ = rows-1;
        if(negx_ < 0) negx_ = 0;
        if(negx_ > cols-1) negx_ = cols-1;
        if(negy_ < 0) negy_ = 0;
        if(negy_ > rows-1) negy_ = rows-1;
        int pos_index = (int)posy_*cols + (int)posx_;
        int neg_index = (int)negy_*cols + (int)negx_;

        O[pos_index] += 1.0;
        O[neg_index] -= 1.0;

        M[pos_index] += mag[index];
        M[neg_index] -= mag[index];
    }
}
for(int r = rstart; r < rend; r++){
    for(int c=cstart; c < cend; c++){
        int index = r*cols + c;
        __TYPE O_ = abs(O[index]);
        if(O_ > kappa) O_ = kappa;

        F[index] = M[index]/kappa * pow(O_/kappa, alpha);
    }
}
Py_END_ALLOW_THREADS
    """ % 'float'
    for r in range(len(radii)):
        n = radii[r]
        # Coordinates of 'positively' and 'negatively' affected pixels
        posx = x + n * imgx
        posy = y + n * imgy
        negx = x - n * imgx
        negy = y - n * imgy
        # Clamp Orientation projection matrix values to a maximum of
        # +/-kappa,  but first set the normalization parameter kappa to the
        # values suggested by Loy and Zelinski
        kappa = 9.9
        if n == 1:
            kappa = 8
        # Form the orientation and magnitude projection matrices
        n_tiles = 1
        tile = 0
        inline(ormagcode, [
            'O',
            'M',
            'mag',
            'posx',
            'posy',
            'negx',
            'negy',
            'kappa',
            'F',
            'alpha',
            'n_tiles',
            'tile',
        ], verbose=0)
        # Generate a Gaussian of size proportional to n to smooth and spread
        # the symmetry measure.  The Gaussian is also scaled in magnitude
        # by n so that large scales do not lose their relative weighting.
        # A = fspecial('gaussian',[n n], 0.25*n) * n;
        # S = S + filter2(A,F);
        width = np.round(0.8 * n)
        if np.mod(width, 2) == 0:
            width += 1
        gauss1d = signal.gaussian(width, 0.25 * n).astype(image.dtype)
        S += separable_convolution2d(F, gauss1d, gauss1d)
    S = S / np.float32(len(radii))
    return S,(mag,imgx,imgy)


def separable_convolution2d(image, row, col, **kwargs):
    ''' Used in sobel and radial transform '''
    code = \
           """
            Py_BEGIN_ALLOW_THREADS

            #define __TYPE  %s

            int h = Nimage[0];
            int w = Nimage[1];

            int image_r_stride = image_array->strides[0];
            int image_c_stride = image_array->strides[1];
            int fp_r_stride = firstpass_array->strides[0];
            int fp_c_stride = firstpass_array->strides[1];

            int row_width = Nrow[0];
            int row_halfwidth;
            if((row_width %% 2) == 0){
                row_halfwidth = (row_width-1) / 2;
            } else {
                row_halfwidth = row_width / 2;
            }

            int col_width = Ncol[0];
            int col_halfwidth;
            if((col_width %% 2) == 0){
                col_halfwidth = (col_width-1) / 2;
            } else {
                col_halfwidth = col_width / 2;
            }

            int r_stride = firstpass_array->strides[0];
            int c_stride = firstpass_array->strides[1];

            // Apply the row kernel
            for(int r = 0; r < h; r++){
                for(int c = 0; c < w; c++){
                    int result_offset = r_stride * r + c_stride*c;
                    __TYPE *result_ptr = (__TYPE *)((char *)firstpass_array->data + result_offset);
                    result_ptr[0] = 0.0;

                    for(int k = 0; k < row_width; k++){
                        int k_index = k - row_halfwidth + c;

                        //if(k_index < 0 || k_index > w) continue;
                        if(k_index < 0) k_index *= -1;  // reflect at boundaries
                        if(k_index >= w) k_index = w - (k - row_halfwidth);

                        __TYPE *image_ptr = (__TYPE *)((char *)image_array->data + image_r_stride*r + image_c_stride*k_index);
                        __TYPE kernel_coef = *((__TYPE*)((char *)row_array->data + row_array->strides[0] * k));
                        *result_ptr += kernel_coef * (*image_ptr);

                    }
                }
            }


            r_stride = result_array->strides[0];
            c_stride = result_array->strides[1];

            // Apply the col kernel
            for(int c = 0; c < w; c++){
                for(int r = 0; r < h; r++){

                    int result_offset = r_stride*r + c_stride*c;
                    __TYPE *result_ptr = (__TYPE *)((char *)result_array->data + result_offset);
                    result_ptr[0] = 0.0;

                    for(int k = 0; k < col_width; k++){
                        int k_index = k - col_halfwidth + r;

                        //if(k_index < 0 || k_index > h) continue;
                        if(k_index < 0) k_index *= -1;  // reflect at boundaries
                        if(k_index >= h) k_index = h - (k - col_halfwidth);

                        __TYPE *image_ptr = (__TYPE *)((char *)firstpass_array->data + k_index*fp_r_stride + fp_c_stride*c);

                        __TYPE kernel_coef = *((__TYPE *)((char *)col_array->data + col_array->strides[0]*k));
                        *result_ptr += kernel_coef * (*image_ptr);

                    }
                }
            }
            Py_END_ALLOW_THREADS
        """ \
            % 'float'

    firstpass = np.zeros_like(image)
    result = np.zeros_like(image)
    inline(code, ['image', 'row', 'col', 'firstpass', 'result'], verbose=0)
    return result


def find_ray_boundaries(im, seed_point, zero_referenced_rays,
                        cutoff_index, threshold,x_axis,y_axis, **kwargs):
    """ Find where a set off rays crosses a threshold in an image

    Arguments:
    im -- An image (usually a gradient magnitude image) in which crossing will be found
    seed_point -- the origin from which rays will be projected
    zero_referenced_rays -- the set of rays (starting at zero) to sample.  nrays x ray_sampling x image_dimension
    cutoff_index -- the index along zero_referenced_rays below which we are sure that we are
    still within the feature.  Used to normalize threshold.
    threshold -- the threshold to cross, expressed in standard deviations across the ray samples
    """

    boundary_points = []

    # create appropriately-centered rays
    rays_x = zero_referenced_rays[:, :, x_axis] + seed_point[x_axis]
    rays_y = zero_referenced_rays[:, :, y_axis] + seed_point[y_axis]

    # get the values from the image at each of the points
    vals = get_image_values(im, rays_x, rays_y)

    cutoff_index = int(cutoff_index)
    if cutoff_index != 0:
        vals_reshaped = vals[:, 0:cutoff_index]
        vals_reshaped = vals_reshaped[np.where(~np.isnan(vals_reshaped))]
        vals_reshaped.shape = [np.prod(vals_reshaped.shape)]
        mean_val = np.mean(vals_reshaped)
        std_val = np.std(vals_reshaped)

        normalized_threshold = threshold * std_val + mean_val
    else:
        normalized_threshold = threshold * np.std(vals[:]) + np.mean(vals[:])

    vals_slope = np.hstack((2 * np.ones([vals.shape[0], 1]), np.diff(vals, 1)))
    vals_slope[np.where(np.isnan(vals_slope))] = 0
    # scan inward-to-outward to find the first threshold crossing
    for r in range(0, vals.shape[0]):
        crossed = False
        for v in range(cutoff_index, vals.shape[1]):
            if np.isnan(v):
                #print "end of ray"
                break
            val = vals[r, v]
#             print "comparing val %f to threshold %f" % (val, normalized_threshold)
            if val > normalized_threshold:
                crossed = True
            if crossed and vals_slope[r, v] <= 0:
                boundary_points.append(np.array([rays_x[r, v - 1], rays_y[r, v - 1]]))
                break
    if 'exclusion_center' in kwargs:
        final_boundary_points = []
        exclusion_center = kwargs['exclusion_center']
        exclusion_radius = kwargs['exclusion_radius']#

        for bp in boundary_points:
            if exclusion_center == None or np.linalg.norm(exclusion_center - bp) > exclusion_radius:
                final_boundary_points.append(bp)
        return final_boundary_points
    else:
        return boundary_points

def pupil_estimate(mag,cr_guess,pupil_guess,
                   cr_ray_length=10, pupil_ray_length=250,
                   cr_min_radius=2, pupil_min_radius=3,
                   cr_n_rays=20, pupil_n_rays=40,
                   cr_ray_sample_spacing = 0.5, pupil_ray_sample_spacing = 1,
                   cr_threshold = 1, pupil_threshold = 2.5,
                   x_axis = 0, y_axis=1): 
    rad_steps = 20
    min_rad_frac = 1./200
    max_rad_frac = 1./5
    cr_ray_sampling = np.arange(0,cr_ray_length,
                                cr_ray_sample_spacing)
    cr_ray_sampling = cr_ray_sampling[1:]
    cr_min_radius_ray_index = np.round(cr_min_radius / cr_ray_sample_spacing)

    pupil_ray_sampling = np.arange(pupil_ray_sample_spacing,
                                   pupil_ray_length,
                                   pupil_ray_sample_spacing)
    pupil_ray_sampling = pupil_ray_sampling[1:]  # don't need the zero sample

    pupil_min_radius_ray_index = np.round(pupil_min_radius / pupil_ray_sample_spacing)
    pupil_ray_sampling = pupil_ray_sampling[1:]  # don't need the zero sample

    cr_rays = np.zeros((cr_n_rays, len(cr_ray_sampling), 2))
    pupil_rays = np.zeros((pupil_n_rays,len(pupil_ray_sampling), 2))

    cr_ray_angles = np.linspace(0, 2 * np.pi, cr_n_rays + 1)
    cr_ray_angles = cr_ray_angles[0:-1]

    pupil_ray_angles = np.linspace(0, 2 * np.pi, pupil_n_rays + 1)
    pupil_ray_angles = pupil_ray_angles[0:-1]

    for r in range(0, cr_n_rays):
        ray_angle = cr_ray_angles[r]
        cr_rays[r, :, x_axis] = cr_ray_sampling * np.cos(ray_angle)
        cr_rays[r, :, y_axis] = cr_ray_sampling * np.sin(ray_angle)

    
    for r in range(0, pupil_n_rays):
        ray_angle = pupil_ray_angles[r]
        pupil_rays[r, :, x_axis] = pupil_ray_sampling * np.cos(ray_angle)
        pupil_rays[r, :, y_axis] = pupil_ray_sampling * np.sin(ray_angle)


# cr!
    cr_boundaries = find_ray_boundaries(mag,
                                        cr_guess, cr_rays,
                                        cr_min_radius_ray_index,
                                        cr_threshold, x_axis, y_axis)
    (cr_position, cr_radius),_ = fitEllipse(cr_boundaries)
    # Pupil
    # do a two-stage starburst fit for the pupil
        # stage 1, rough cut
    pupil_boundaries = find_ray_boundaries(
        mag,
        pupil_guess,
        pupil_rays,
        pupil_min_radius_ray_index,
        pupil_threshold,
        x_axis,
        y_axis,
        exclusion_center=np.array(cr_position),
        exclusion_radius=2 * cr_radius[0],
    )
    (pupil_position, pupil_radius),_ = fitEllipse(pupil_boundaries)

    # stage 2: refine
    minimum_pupil_guess = np.round(0.5 * pupil_radius[0] / pupil_ray_sample_spacing)
    pupil_boundaries = find_ray_boundaries(
        mag,
        pupil_position,
        pupil_rays,
        minimum_pupil_guess,
        pupil_threshold,
        x_axis,
        y_axis,
        exclusion_center=np.array(cr_position),
        exclusion_radius=2 * cr_radius[0],
    )
    (pupil_position, pupil_radius), pupil_ellipse_par = fitEllipse(pupil_boundaries)

    return (cr_position, cr_radius),(pupil_position, pupil_radius),pupil_ellipse_par

def get_image_values(im, x_, y_):
    """ Sample an image at a set of x and y coordinates, using nearest neighbor interpolation.
    """

    x = x_.round()
    y = y_.round()

    # trim out-of-bounds elements
    bad_elements = np.where((x < 0) | (x >= im.shape[0]) | (y < 0) | (y >= im.shape[1]))

    x[bad_elements] = 0
    y[bad_elements] = 0

    vals = im[x.astype(int), y.astype(int)]
    vals[bad_elements] = np.nan
    return vals

def find_minmax(image, **kwargs):
    '''Deprecated?use cv2.'''
    if image == None:
        return ([0, 0], [0])

    code = \
           """
            Py_BEGIN_ALLOW_THREADS


            int rows = Nimage[0];
            int cols = Nimage[1];

            #define __TYPE  %s

            __TYPE themax = -999999;
            __TYPE themin = 999999;

            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){

                    __TYPE *pixel_ptr = (__TYPE *)((char *)image_array->data + r * image_array->strides[0] + c * image_array->strides[1]);


                    if(*pixel_ptr > themax){

                        themax = *pixel_ptr;
                        coordinates[2] = (__TYPE)r;
                        coordinates[3] = (__TYPE)c;
                    }

                    if(*pixel_ptr < themin){

                        themin = *pixel_ptr;
                        coordinates[0] = (__TYPE)r;
                        coordinates[1] = (__TYPE)c;
                    }
                }
            }

            Py_END_ALLOW_THREADS
        """ \
            % 'float'
    coordinates = np.array([0., 0., 0., 0.])
    themax = 0.
    themin = 0.

    inline(code, ['image', 'coordinates'])
    return (coordinates[0:2], coordinates[2:4])
