
import astropy.units as u
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
    USE_OPENCV=True
except:
    USE_OPENCV=False
    import scipy.ndimage

def rotate(img,angle):
        n_y,n_x=img.shape
        if USE_OPENCV:
                print("Using openCV for rotations" )
                return cv2.warpAffine(img,
                                  cv2.getRotationMatrix2D((n_y/2,n_x/2),angle,1),
                                  (n_y,n_x),
                                  flags=cv2.INTER_AREA)
        else:
                return scipy.ndimage.rotate(img,angle,reshape=False,order=3)

def fit_to_hlc(zodi_file,
               interpolating_function,
               xmas,
               HLC_plate_scale,
               n = 200, # size of arrays from Krist so make this the default
               thresh=10,
               display=False,
               core_mask_radius=0*u.arcsec,
               mask_radius=None,
               vmax_out=None,
              **kwargs):
    
    print("Running zodipic through HLC ...")
    
    print('\nInput file: ')
    print(zodi_file)
    
    zodi_fits = fits.open(zodi_file)
    
    index = zodi_fits[0].data < zodi_fits[0].data.max()/thresh
    zodi = np.ma.masked_array(zodi_fits[0].data,index)
    # creates the masked array of the zodi data based on if the zodi value in a pixel is less than the max divided by the thresh
    
    zodi_pixscale = zodi_fits[0].header["PIXELSCL"]*u.arcsecond # gets the pixel scale of the zodi file from its header
    
    pixnum = np.int(zodi_fits[0].data.shape[0])
    xpix,ypix = np.meshgrid(np.arange(-pixnum/2,pixnum/2),np.arange(-pixnum/2,pixnum/2))
    # creates a 256by256 grid ranging from -128 to 127
    
    x = (xpix+.5).flatten()*zodi_pixscale
    y = (ypix+.5).flatten()*zodi_pixscale
    xcenter = (pixnum/2+.5)*zodi_pixscale
    ycenter = (pixnum/2+.5)*zodi_pixscale
    # Because there are an even number of pixels, do halfpixel shifts to center x,y grid and multiply by the pixel scale
    # to create the x,y grid in arcseconds
    
    index[(np.sqrt((x)**2 + (y)**2)<core_mask_radius).reshape([pixnum,pixnum])]=True
    # adjusts the masked array based on the core mask radius
    
    xpix_flat = xpix.flatten()
    ypix_flat = ypix.flatten()

    im = np.zeros([n,n])
    # flattens the pixel grid and initializes the im which will be 200 by 200
    
    for i,zodi_val in enumerate(zodi.flatten()):

        if np.ma.is_masked(zodi_val):
            continue
        im += zodi_val*closest_monochrome_PSF(x[i],y[i],
                                              interpolating_function,
                                              xmas,
                                              HLC_plate_scale,
                                              n = n,
                                              mask_radius=mask_radius,
                                             **kwargs)
    '''
    this for loop runs through the data in the zodi masked array. 
    if the value is masked, then the loop does nothing and continues to the next value. 
    if the value is not masked, then the loop creates an im using the zodi_val as a weight and by calling the 
    closest_monochrome_psf function with the corresponding x,y to that zodi_val. 
    Each time a loop is completed, the im summed up again.
    '''
    if display:
        plt.close()
        plt.figure(figsize = [11,3])
        
        ax = plt.subplot(1, 3, 1)
        zodiplot=plt.imshow(zodi)
        ax.set_title("Input Zodi")
        ax.set_ylabel("pixels")
        plt.colorbar(zodiplot)
        
        ax = plt.subplot(1, 3, 2)
        maskplot = ax.imshow(index,cmap=plt.cm.bone, vmin=0, vmax=1)
        ax.set_title("Mask")
        plt.colorbar(maskplot)
        ax.set_xlabel("pixels")

        ax = plt.subplot(1, 3, 3)
        if vmax_out==None:
            vmax_out=im.max()
        implot=ax.imshow(im,vmax=vmax_out,)

        ax.set_title('Zodi through HLC',)
        plt.colorbar(implot)
        ax.set_xlabel("pixels")

    return im.T,zodi
   

def closest_monochrome_PSF(x,y,
                           HLCinterpfun,
                           xmas,
                           HLC_plate_scale,
                           n=200, # size of arrays from Krist so make this the default
                           mask_radius=None):
    
    '''
    Returns the nearest PSF realization by interpolating across grid of input 
    
    '''
    
    r,theta = tilt_converter(x,y)
    
    grid=np.meshgrid(range(n),range(n))
    flattened_grid = np.vstack([grid[0].flatten(),grid[1].flatten()])
    mask=1
    
    if mask_radius is not None:
        if mask_radius > n:
            raise ValueError("Mask radius exceeds array size")
        xp,yp=((+n/2-x/HLC_plate_scale).decompose(),(y/HLC_plate_scale).decompose()+n/2)
        mask=np.sqrt((-grid[0]+yp)**2+(grid[1]-xp)**2)<=mask_radius
    
    r=r.to(u.milliarcsecond)
    
    pts = np.vstack([ flattened_grid , r.value*np.ones(len(grid[0].flatten())) ]).T
    
    interpped = HLCinterpfun(pts).reshape(n,n)
    
    interpped = rotate(interpped,-theta.to(u.deg).value)*mask
    
    return interpped

## Calculate the theta/r value -- This is annoying since the wavefront is in-xy space.
## but the wfirst CGI doesn't inherit OpticalSystem, so I don't see an easy way to redefine 
## input_wavefront() to use xy coordinates

# These functions should move to an external .py module once done debugging.

def tilt_converter(x,y):
    """
    converts from xy coordinates to angle and rotation angle

    Parameters
    ----------  
    x: float
     pixel position
    y: float
     pixel position

    
    Returns
    ----------  

    (r,theta):
            where r is in the units of x and y and theta is in degrees
    
    """
    return (np.sqrt(x**2+y**2), np.arctan2(-x,y))*u.deg

def cartesian_off_axis_psf_poppy(cgi,x,y,wavels,**kwargs):
    r,theta = tilt_converter(x,y)
    #print(x,y,r,theta)
    ifs_spc.options['source_offset_r'] = r # arcsec
    ifs_spc.options['source_offset_theta'] = theta # deg w.r.t. North
    ifs_psf = ifs_spc.calc_datacube(wavels, **kwargs)
    return ifs_psf

import scipy.ndimage
import scipy.interpolate
def cartesian_off_axis_psf_interpol(cgi,x,y,wavels,**kwargs):
    r,theta = tilt_converter(x,y)
    PSF = fits.open()
    return ifs_psf
