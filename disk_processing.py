from importlib import reload
from pathlib import Path
import hlc_processing
reload(hlc_processing)

import astropy.units as u
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import os
my_home_dir = os.getcwd()

def disk_through_hlc(fitsfile,
                     my_interp_fun,
                     xmas,
                     HLC_plate_scale_AS,
                     n = 200,
                     thresh=1000,
                     display=False,
                     x_extent=0.5,y_extent=0.5,
                     load_existing=False,
                     localzodi=0):
    
    if fitsfile.find("89") !=-1:
        thresh=50000
    
    print("Running disk through HLC ...")
    
    my_home_path = Path(os.getcwd())
    debes_path = my_home_path/'DebesModels'
    
    debes_out_path = my_home_path/'DebesModels_outputs'
    if "annulus" in fitsfile:
        fits_path = debes_path/"Model_Annulus"/fitsfile
        out_path = debes_out_path/"Model_Annulus"/(fitsfile[:-5]+"_HLC_"+str(thresh)+'.fits')
    if "constanttau" in fitsfile:
        fits_path = debes_path/"Model_ConstantTau"/fitsfile
        out_path = debes_out_path/"Model_ConstantTau"/(fitsfile[:-5]+"_HLC_"+str(thresh)+'.fits')
    if "gap" in fitsfile:
        fits_path = debes_path/"Model_GAP"/fitsfile
        out_path = debes_out_path/"Model_GAP"/(fitsfile[:-5]+"_HLC_"+str(thresh)+'.fits')
    if "ring" in fitsfile:
        fits_path = debes_path/"Model_Ring"/fitsfile
        out_path = debes_out_path/"Model_Ring"/(fitsfile[:-5]+"_HLC_"+str(thresh)+'.fits')
    
    zodi_fits = fits.open(fits_path)
    
    print("\nInput file: ")
    print(fits_path)
    print("\nOutput file: ")
    print(out_path)

    print("\nThresh = " + str(thresh))
    print("local zodi: {}".format(localzodi))

    index = zodi_fits[0].data < zodi_fits[0].data.max()/thresh
    zodi = np.ma.masked_array(zodi_fits[0].data, index)
    # creates the masked array of the zodi data based on if the zodi value in a pixel is less than the max divided by the thresh
    
    zodi_pixscale = zodi_fits[0].header["PIXELSCL"]*u.arcsecond # gets the pixel scale of the zodi file from its header

    pixnum = np.int(zodi_fits[0].data.shape[0])
    x,y = np.meshgrid(np.arange(-pixnum/2,pixnum/2),np.arange(-pixnum/2,pixnum/2))
    x = (x+.5).flatten()*zodi_pixscale
    y = (y+.5).flatten()*zodi_pixscale
    # creates an x,y grid with the x,y in arcseconds based on the pixel scale of the fits file
    
    im = np.zeros([n,n])
    
    try:
        if load_existing:
            print("\nload_existing keyword True.")
            im = fits.getdata(outfits)
        else:
            print("\nload_existing keyword False, throwing error to regenerate file.")
            raise ValueError("")

    except:
        for i,zodi_val in enumerate(zodi.flatten()):
            if np.ma.is_masked(zodi_val):
                continue
            im += (localzodi+zodi_val)*hlc_processing.closest_monochrome_PSF(x[i],y[i],
                                                                             my_interp_fun,
                                                                             xmas,
                                                                             HLC_plate_scale_AS,
                                                                             n=n)
        
    if display:
        
        plt.figure(figsize=[9,4])
        
        plt.subplot(121)
        halfpix = zodi_pixscale.to(u.arcsec).value*0.5
        extent = ([x.min().to(u.arcsec).value-halfpix,
                   x.max().to(u.arcsec).value+halfpix, 
                   y.min().to(u.arcsec).value-halfpix,
                   y.max().to(u.arcsec).value+halfpix])
        plt.title("Input Flux")
        plt.xlim([-x_extent,x_extent])
        plt.ylim([-y_extent,y_extent])
        plt.ylabel("$\prime\prime$")
        plt.xlabel("$\prime\prime$")
        plt.grid()
        plt.imshow(zodi,extent=extent,norm=LogNorm(zodi.data[zodi>0].min(),zodi.data.max()))#zodi.data.min(),zodi.data.max))
        axc = plt.colorbar()
        axc.set_label("Jy")
        
        plt.subplot(122)
        pixnum = np.int(im.shape[0])
        x,y = np.meshgrid(np.arange(-pixnum/2,pixnum/2),np.arange(-pixnum/2,pixnum/2))
        x = (x+.5).flatten()*HLC_plate_scale_AS
        y = (y+.5).flatten()*HLC_plate_scale_AS
        halfpix = .5*HLC_plate_scale_AS.value
        extent = ([x.min().to(u.arcsec).value-halfpix,
                   x.max().to(u.arcsec).value+halfpix, 
                   y.min().to(u.arcsec).value-halfpix,
                   y.max().to(u.arcsec).value+halfpix])
        plt.imshow(im,extent=extent)
        plt.colorbar()
        plt.title("HLC Image of Included Flux")
        plt.xlim([-x_extent,x_extent])
        plt.ylim([-y_extent,y_extent])
        plt.grid()
        plt.xlabel("$\prime\prime$")
        plt.tight_layout()
        
        plt.show()
        
    return im,zodi_fits,out_path
