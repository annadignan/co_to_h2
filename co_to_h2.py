# import statements

# astropy
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.table import QTable, Table, Column, vstack
from astropy import constants as const
from astropy.coordinates import SkyCoord

from photutils.aperture import SkyCircularAperture, ApertureStats

from reproject import reproject_interp

# this assumes you have CO_conversion_factor installed from Jiayi Sun
# !pip install CO-conversion-factor
from CO_conversion_factor import metallicity

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# general / misc
import numpy as np

# defining helper functions
def plotmap(data_in, clabel=None, norm=False, cmap='viridis', title=None, savepath=''):

    """
    Displays an image with customizable options for clarity.

    Parameters
    ----------
    data_in : numpy.ndarray
        Image to be displayed.
    clabel: str, optional
        Colorbar label to be displayed.
    norm : bool, optional
        Whether to use norm=LogNorm() for stretch. Default is False. 
    cmap : str, optional
        Colormap to use. Default is viridis.
    title : str, optional
        Title to be displayed at top of plot.
    savepath: str, optional
        If specified, path for saving resulting figure. 

    Returns
    -------
    None
        This function produces a plot and does not return a value.
        """

    plt.figure()
    if norm:
        plt.imshow(data_in, norm=LogNorm(), cmap=cmap)
    else:
        plt.imshow(data_in, cmap=cmap)
    if title:
        plt.title(title)
    if clabel:
        plt.colorbar(label=clabel)
    if clabel==None:
        plt.colorbar()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

def photometry(data, ras_in, decs_in, radius, wcs_file, method, deg=True, reg_name=None):

    """
    Performs aperture photometry on an input image.

    Parameters
    ----------
    data : numpy.ndarray
        Image on which photometry will be performed.
    ras_in: list or numpy.ndarray
        Right ascension values.
    decs_in: list or numpy.ndarray
        Declination values.
    radius : astropy.units.Quantity
        Aperture radius with units.
    wcs_file : str
        Path to FITS file with desired WCS. 
    method : str
        Method used for deriving mass to light ratio.
    deg : bool
        Whether the input coordinates are already in decimal degree format. Default is True.
    reg_name: str, optional
        Name of aperture region. 

    Returns
    -------
    astropy.table.QTable
        Table containing input coordinates, aperture radii, 
        and measured aperture sum for each method.
        """

    # initialize photometry table
    phot_tab=QTable()
    # also initialize list for photometry results
    sums=[]

    # get WCS
    data, header = fits.getdata(wcs_file, ext=0, header=True)
    wcs = WCS(header)

    # make sure the coordinates and radii are lists
    # this allows us to iterate over the radius as a Quantity
    # and iterate over all three at once
    if np.isscalar(ras_in):
        ras_in = [ras_in]

    if np.isscalar(decs_in):
        decs_in = [decs_in]

    if getattr(radius, 'isscalar', False):
        radius = [radius]

    # format the coords if not already in decimal degrees
    for r, d, rad in zip(ras_in, decs_in, radius):
        
        if deg==False:
            # this assumes hh:mm:ss.s format
            r_split=r.split(':')
            ra=(r_split[0]+'h'+r_split[1]+'m'+r_split[2]+'s')

            # this assumes dd:mm:ss.s format
            d_split=d.split(':')
            dec=(d_split[0]+'d'+d_split[1]+'m'+d_split[2]+'s')
        # if in decimal degree format, add units
        else:
            ra=r*u.deg
            dec=d*u.deg

        # make sure the radius has units (required for SkyCircularAperture)
        if not hasattr(rad, 'unit'):
            raise ValueError("Please give units with the radius.")

        # create apertures from inputs
        pos=SkyCoord(ra, dec)
        aperture=SkyCircularAperture(pos, rad)

        # perform photometry and save results
        aperstats= ApertureStats(data=data, aperture=aperture, wcs=wcs)
        sums.append(aperstats.sum)

    # create new columns for saving aperture info (inputs and output)
    ras = Column(name='RA_J2000', data=ras_in)
    decs = Column(name='Decl_J2000', data=decs_in)
    radius = u.Quantity(radius)
    radii = Column(name='radius', data=radius)
    sum = Column(name=f'aperture sum ({method})', data=sums)

    # also save the region name as its own column
    if reg_name:
        name = Column(name='region name', data=[reg_name]*len(sums))
        phot_tab.add_columns([ras, decs, radii, sum, name])
    if not reg_name:
        phot_tab.add_columns([ras, decs, radii, sum])
    
    return phot_tab

# defining main Class and methods
class Map:
    '''
    This class loads in a FITS file and sets it up for performing operations
    such as reprojection and calculating various astrophysical quantities.
    Note that a FITS file path is not required; a Map can be created from
    an output of a method of the Map class.

    Attributes
    ----------
    data : numpy.ndarray
        Image data array extracted from the FITS file.

    hdr : astropy.io.fits.Header
        FITS header associated with the image.

    wcs : astropy.wcs.WCS
        World Coordinate System object extracted from the header.

    unit : astropy.units.Unit
        Physical units of the image data (from BUNIT keyword if header is specified).
    '''

    def __init__(self, path=None, data=None, header=None, unit=None, wcs=None):
        '''Constructs all the necessary attributes for the Map object.'''
        # extract data array, header, and WCS from FITS file if specified
        if path is not None:
            data, header = fits.getdata(path, ext=0, header=True)
            self.hdr = header
            self.wcs = WCS(self.hdr)

        # split into attributes 
        self.data = data
        self.hdr = header

        # if units are given as an argument, assign them
        if unit:
            self.unit = unit
        # if BUNIT keyword exists, assign value as unit
        elif header is not None and header['BUNIT'] is not None:
            # special handling for weirdly formatted WISE files
            if header['BUNIT']=='MJY/SR':
                unit=u.MJy/u.sr 
            else:
                unit=u.Unit(header['BUNIT'])
            self.unit = unit
        # if no units are specified, assign dimensionless units
        if unit is None:
            self.unit = u.dimensionless_unscaled

    def reproject(self, template_file, plot=False, title=None, template_unit=None):
        '''Reprojects an input FITS file onto the WCS and size of the 
        input template FITS file.

        Parameters
        ----------
        template_file: str
            Path to FITS file to be reprojected.
        plot: bool
            Whether to plot the output using plotmap(). Default is False.
        title: str, optional
            Title for plotmap.
        template_unit: str, optional
            Units for template FITS file.

        Returns
        -------
        astropy.table.Quantity
            Reprojected data array with units of reprojected FITS file.
        '''

        # create an instance of the input template FITS file
        if template_unit is None:
            template=Map(template_file)
        if template_unit is not None:
            template=Map(template_file, unit=template_unit)

        # remove all axes with length=1 so array is 2D
        data_in = np.squeeze(self.data)

        # get WCS for file to be reprojected and template 
        wcs_in=WCS(self.hdr)
        wcs_out=WCS(template.hdr)

        # get size for reprojection from header of template
        ny=template.hdr['NAXIS2']
        nx=template.hdr['NAXIS1']

        # reproject using reproject_interp()
        data_out, footprint = reproject_interp((data_in, wcs_in), wcs_out,
                                               shape_out=(ny, nx))
        # replace any values outside of original image with NaN
        data_out[footprint <= 0] = np.nan

        if plot == True:
            plotmap(data_out, norm=True, clabel=str(self.unit), title=title)

        return Map(data=data_out, header=self.hdr)

    def add_col(self, t, name, values=None, length=None):

        '''Adds a column to an astropy Table.

        Parameters
        ----------
        t: astropy.table.QTable
            Table to modify.
        name: str
            Name of column to add.
        values: float, numpy.ndarray, or astropy.table.Quantity
            Elements to add to column.
        length: int
            Desired length of column.

        Returns
        -------
        astropy.table.QTable
            Modified Table.
        '''

        # if no input data is specified, extract it from the Map object
        if values is None:
            values=self.data
            unit = self.unit
        # if input data is specified, extract units from input data
        # we want to make sure our QTable has units!!!
        else:
            if hasattr(values, 'unit'):
                unit=values.unit
                values=values.value
        
        # this allows a user to extend a float to a desired length
        if length:
            t[name] = np.full(length, values) * unit
        # if length isn't specified, make sure data is a 1D array for storing
        else:
            t[name] = np.array(values).reshape(-1) * unit
        return t[name]
    
    def calc_upsilon(self, method='', gal_sfr = None, gal_Mstar = None,
               I_w1 = None, I_w3 = None, I_w4 = None):
        
        '''Calculates the mass to light ratio based on the prescriptions
        given in Table 6 and using Equation 24 of Leroy (2019).

        Parameters
        ----------
        method: str
            Name of desired method to calculate mass to light ratio.
        gal_sfr: float
            Log of galaxy's star formation rate in solar masses per year.
        gal_Mstar: float
            Log of galaxy's stellar mass in solar masses.
        I_w1: astropy.units.Quantity
            WISE Band 1 intensity map.
            Required for all methods.
        I_w3: astropy.units.Quantity
            WISE Band 3 intensity map.
            Required for method='w3w1'.
        I_w4: astropy.units.Quantity
            WISE Band 4 intensity map.
            Required for method='w4w1'.

        Returns
        -------
        astropy.units.Quantity
            Mass-to-light ratio (Upsilon) calculated using the specified method.
            Units are solar mass per solar luminosity (M_sun / L_sun).

        Notes
        -----
        The result is also stored as an attribute of the object, with name
        depending on the method:

        - ups_gswlc
        - ups_w3w1
        - ups_w4w1
        '''

        # GSWLC specific star formation rate
        if method == 'gswlc':
            a = -10.9
            b = -0.21
            c = -9.5
            gal_sfr = gal_sfr * u.M_sun / u.yr
            gal_Mstar = gal_Mstar * u.M_sun
            q=np.log10(gal_sfr.value/gal_Mstar.value)
            
        # WISE3-to-WISE1 color
        if method == 'w3w1':
            a = 0.1
            b = -0.46
            c = 0.75
            q=np.log10(I_w3.value/I_w1.value)

        # WISE4-to-WISE1 color
        if method == 'w4w1':
            a = 0.0
            b = -0.4
            c = 0.75
            q=np.log10(I_w4/I_w1).value

        # cap upsilon at high and low values per Equation 24 of Leroy (2019)
        upsilon = np.where(q < a, 0.5,
                np.where(q > c, 0.2,
                0.5+b*(q - a)))
        
        if method == 'gswlc':
            attr_name = 'ups_gswlc'
        if method == 'w3w1':
            attr_name = 'ups_w3w1'
        if method == 'w4w1':
            attr_name = 'ups_w4w1'

        setattr(self, attr_name, upsilon * u.M_sun / u.L_sun)

        return getattr(self, attr_name)
    
    def calc_sig_star(self, upsilon, I_w1, i, method, plot=False):
        '''Calculates the stellar mass surface density based on 
        Equation 2 of Leroy (2021).

        Parameters
        ----------
        upsilon: astropy.units.Quantity
            Mass to light ratio in solar mass per solar luminosity.
        I_w1: astropy.units.Quantity
            WISE Band 1 intensity map.
        i: float
            Galaxy's inclination angle in degrees.
        method: str
            Method used for calculating mass to light ratio.
        plot: bool, optional
            Whether to plot the result. Default is False.

        Returns
        -------
        astropy.units.Quantity
            Stellar mass surface density (sig_star).
            Units are solar masses per parsec squared. 

        Notes
        -----
        The result is also stored as an attribute of the object, with name
        depending on the method:

        - sigstar_gswlc
        - sigstar_w3w1
        - sigstar_w4w1
        '''

        # Equation 2 from Leroy (2021)
        sig_star = 330 * (upsilon/0.5) * I_w1/(u.MJy/u.sr) * np.cos(np.deg2rad(i))

        attr_name = f'sigstar_{method}'

        if plot == True:
            plotmap(sig_star.data, norm=True, clabel=r'M$_\odot$/pc$^{2}$', cmap='inferno', 
                    title=rf'$\Sigma_{{\star}}: \mathrm{{{method}}}$')
    
        setattr(self, attr_name, sig_star.value * u.M_sun / u.pc**2)
        return getattr(self, attr_name)

    # Code credit: Adam Leroy (2025)
    # https://github.com/akleroy/aklpyutils/utils_deproject.py
    def deproject(self, center_coord=None, incl=0*u.deg, pa=0*u.deg,
                header=None, wcs=None, naxis=None, ra=None, dec=None,
                return_offset=False, linear=True, distance=None):

        """
        Calculate deprojected radii and projected angles in a disk.

        This function deals with projected images of astronomical objects
        with an intrinsic disk geometry. Given sky coordinates of the
        disk center, disk inclination and position angle, this function
        calculates deprojected radii and projected angles based on
        (1) a FITS header (`header`), or
        (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
        (3) RA and DEC coodinates (`ra` + `dec`).
        Both deprojected radii and projected angles are defined relative
        to the center in the inclined disk frame. For (1) and (2), the
        outputs are 2D images; for (3), the outputs are arrays with shapes
        matching the broadcasted shape of `ra` and `dec`.

        Parameters
        ----------
        center_coord : `~astropy.coordinates.SkyCoord` object or 2-tuple
            Sky coordinates of the disk center
        incl : `~astropy.units.Quantity` object or number, optional
            Inclination angle of the disk (0 degree means face-on)
            Default is 0 degree.
        pa : `~astropy.units.Quantity` object or number, optional
            Position angle of the disk (red/receding side, North->East)
            Default is 0 degree.
        header : `~astropy.io.fits.Header` object, optional
            FITS header specifying the WCS and size of the output 2D maps
        wcs : `~astropy.wcs.WCS` object, optional
            WCS of the output 2D maps
        naxis : array-like (with two elements), optional
            Size of the output 2D maps
        ra : array-like, optional
            RA coordinate of the sky locations of interest
        dec : array-like, optional
            DEC coordinate of the sky locations of interest
        return_offset : bool, optional
            Whether to return the angular offset coordinates together with
            deprojected radii and angles. Default is to not return.

        Returns
        -------
        deprojected coordinates : list of arrays
            If `return_offset` is set to True, the returned arrays include
            deprojected radii, projected angles, as well as angular offset
            coordinates along East-West and North-South direction;
            otherwise only the former two arrays will be returned.

        Notes
        -----
        This is the Python version of an IDL function `deproject` included
        in the `cpropstoo` package. See URL below:
        https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
        """

        if isinstance(center_coord, SkyCoord):
            x0_deg = center_coord.ra.degree
            y0_deg = center_coord.dec.degree
        else:
            x0_deg, y0_deg = center_coord
            if hasattr(x0_deg, 'unit'):
                x0_deg = x0_deg.to(u.deg).value
                y0_deg = y0_deg.to(u.deg).value
        if hasattr(incl, 'unit'):
            incl_deg = incl.to(u.deg).value
        else:
            incl_deg = incl
        if hasattr(pa, 'unit'):
            pa_deg = pa.to(u.deg).value
        else:
            pa_deg = pa

        if header is not None:
            wcs_cel = WCS(header).celestial
            naxis1 = header['NAXIS1']
            naxis2 = header['NAXIS2']
            # create ra and dec grids
            ix = np.arange(naxis1)
            iy = np.arange(naxis2).reshape(-1, 1)
            ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
        elif (wcs is not None) and (naxis is not None):
            wcs_cel = wcs.celestial
            naxis1, naxis2 = naxis
            # create ra and dec grids
            ix = np.arange(naxis1)
            iy = np.arange(naxis2).reshape(-1, 1)
            ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
        else:
            ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
            if hasattr(ra_deg, 'unit'):
                ra_deg = ra_deg.to(u.deg).value
                dec_deg = dec_deg.to(u.deg).value

        # recast the ra and dec arrays in term of the center coordinates
        # arrays are now in degrees from the center
        dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
        dy_deg = dec_deg - y0_deg

        # rotation angle (rotate x-axis up to the major axis)
        rotangle = np.pi/2 - np.deg2rad(pa_deg)

        # create deprojected coordinate grids
        deprojdx_deg = (dx_deg * np.cos(rotangle) +
                        dy_deg * np.sin(rotangle))
        deprojdy_deg = (dy_deg * np.cos(rotangle) -
                        dx_deg * np.sin(rotangle))
        deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

        # make map of deprojected distance from the center
        radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

        # make map of angle w.r.t. position angle
        projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

        self.projang_deg=projang_deg
        self.dx_deg=dx_deg
        self.dy_deg=dy_deg

        if linear==True and distance is None:
            raise ValueError("Please provide a distance with units.")
        if linear==True and distance is not None:
            if hasattr(distance, 'unit'):
                distance_kpc=(distance).to(u.kpc).value 
            if not hasattr(distance, 'unit'):
                raise ValueError("Please give units with the distance.")
            
            r_g = radius_deg * u.deg.to(u.radian) * distance_kpc
            r_g_kpc = r_g * u.kpc

            self.radius_kpc=r_g_kpc
            return self.radius_kpc

        if linear==False:

            self.radius_deg=radius_deg

            if return_offset:
                return self.radius_deg, self.projang_deg, self.dx_deg, self.dy_deg
            else:
                return self.radius_deg, self.projang_deg
        
    # Code credit: Jiayi Sun (2020)
    # https://github.com/astrojysun/COConversionFactor/CO_conversion_factor/metallicity.py
    
    # this function is needed for the calc_metallicity function called later
    def predict_logOH_SAMI19(self,
            Mstar, calibration='PP04', form='MZR', return_residual=False):
        """
        Predict 12+log(O/H) with the 'SAMI19' MZR (Sanchez+19).

        This function predicts the gas phase abundance 12+log(O/H)
        from the global stellar mass of a galaxy according to the
        mass-metallicity relations reported in Sanchez+19.
        Reference: Sanchez et al. (2019), MNRAS, 484, 3042

        Parameters
        ----------
        Mstar : number, `~numpy.ndarray`, `~astropy.units.Quantity` object
            Galaxy global stellar mass, in units of solar mass
        calibration : {'O3N2-M13', 'PP04', 'N2-M13', 'ONS', 'R23', ...}
            Metallicity calibration to adopt (see Table 1 in Sanchez+19).
            Default is 'PP04'.
        form : {'MZR', 'pMZR'}
            The MZR functional form to adopt (see Table 1 in Sanchez+19).
            Default is 'MZR'.
        return_residual : bool
            Whether to return the residual scatter around the MZR.
            Default is to not return.

        Returns
        -------
        logOH : number or `~numpy.ndarray`
            Predicted gas phase abundance, in units of 12+log(O/H)
        """

        calibrations = [
            'O3N2-M13', 'PP04', 'N2-M13', 'ONS', 'R23', 'pyqz', 't2',
            'M08', 'T04', 'EPM09', 'DOP16']
        if calibration not in calibrations:
            raise ValueError(
                "Available choices for `calibration` are: "
                "{}".format(calibrations))

        if hasattr(Mstar, 'unit'):
            x = np.log10(Mstar.to(u.Msun).value) - 8
        else:
            x = np.log10(Mstar) - 8

        # Mass-metallicity relation
        if form == 'MZR':  # MZR best fit
            params = np.zeros(
                3, dtype=[(calib, 'f') for calib in calibrations])
            params[0] = (  # a
                8.51, 8.73, 8.50, 8.51, 8.48, 9.01, 8.84,
                8.88, 8.84, 8.54, 8.94)
            params[1] = (  # b
                0.007, 0.010, 0.008, 0.011, 0.004, 0.017, 0.008,
                0.010, 0.007, 0.002, 0.020)
            params[2] = (  # sigma_MZR
                0.102, 0.147, 0.105, 0.138, 0.101, 0.211, 0.115,
                0.169, 0.146, 0.074, 0.288)
            a = params[calibration][0]
            b = params[calibration][1]
            c = 3.5
            residual = params[calibration][2]
            logOH = a + b*(x-c)*np.exp(-(x-c))

        elif form == 'pMZR':  # pMZR polynomial fit
            params = np.zeros(
                5, dtype=[(calib, 'f') for calib in calibrations])
            params[0] = (  # p0
                8.478, 8.707, 8.251, 8.250, 8.642, 8.647, 8.720,
                8.524, 8.691, 8.456, 8.666)
            params[1] = (  # p1
                -0.529, -0.797, -0.207, -0.428, -0.589, -0.718, -0.487,
                -0.148, -0.200, -0.097, -0.991)
            params[2] = (  # p2
                0.409, 0.610, 0.243, 0.427, 0.370, 0.682, 0.415,
                0.218, 0.164, 0.130, 0.738)
            params[3] = (  # p3
                -0.076, -0.113, -0.048, -0.086, -0.063, -0.133, -0.080,
                -0.040, -0.023, -0.032, -0.114)
            params[4] = (  # sigma_pMZR
                0.077, 0.112, 0.078, 0.101, 0.087, 0.143, 0.087,
                0.146, 0.123, 0.071, 0.207)
            p0 = params[calibration][0]
            p1 = params[calibration][1]
            p2 = params[calibration][2]
            p3 = params[calibration][3]
            residual = params[calibration][4]
            logOH = p0 + p1*x + p2*x**2 + p3*x**3

        else:
            raise ValueError("Invalid input value for `form`!")

        self.logOH=logOH
        self.residual=residual

        if return_residual:
            return self.logOH, self.residual
        else:
            return self.logOH

    # Code credit: Jiayi Sun (2020)
    # https://github.com/astrojysun/COConversionFactor/CO_conversion_factor/metallicity.py
    
    # this function also needed for calc_metallicity later
    def extrapolate_logOH_radially(self,
            logOH_Re, gradient='CALIFA14', Rgal=None, Re=None):
        """
        Extrapolate 12+log(O/H) assuming a fixed radial gradient.

        This function extrapolates the gas phase abundance 12+log(O/H)
        from its value at 1 Re to the entire galaxy, according to a fixed
        radial gradient specified by the user.

        Parameters
        ----------
        logOH_Re : number, `~numpy.ndarray`
            Gas phase abundance at 1 Re, in units of 12+log(O/H)
        gradient : {'CALIFA14', float}
            Radial abundance gradient to adopt, in units of dex/Re.
            Default is 'CALIFA14', i.e., -0.10 dex/Re
            (Reference: Sanchez et al. 2014, A&A, 563, A49).
        Rgal : number, ndarray, Quantity object
            Galactocentric radii, in units of kilo-parsec
        Re : number, ndarray, Quantity object
            Galaxy effective radii, in units of kilo-parsec

        Returns
        -------
        logOH : number or `~numpy.ndarray`
            Predicted gas phase abundance, in units of 12+log(O/H)
        """

        if (Rgal is None) or (Re is None):
            return logOH_Re
        
        if hasattr(Rgal, 'unit') and hasattr(Re, 'unit'):
            Rgal_normalized = (Rgal / Re).to('').value
        elif hasattr(Rgal, 'unit') or hasattr(Re, 'unit'):
            raise ValueError(
                "`Rgal` and `Re` should both carry units "
                "or both be dimensionless")
        else:
            Rgal_normalized = np.asarray(Rgal) / np.asarray(Re)

        # metallicity gradient
        if gradient == 'CALIFA14':
            alpha_logOH = -0.10  # dex/Re
            logOH = (logOH_Re + alpha_logOH * (Rgal_normalized - 1))
        else:
            alpha_logOH = gradient  # dex/Re
            logOH = (logOH_Re + alpha_logOH * (Rgal_normalized - 1))

        self.logOH=logOH
        return self.logOH

    def calc_metallicity(self, distance, Mstar, Re, r_gal,
                          unit='', logOH_solar=None):
        """ 
        Calculate the metallicity based on the method specified.

        Parameters:
        ----------
        distance: astropy.units.Quantity
            Distance to galaxy. 
        Mstar : astropy.units.Quantity
            Log of galaxy's stellar mass in solar units.
        Re: astropy.units.Quantity
            Galaxy's effective radius.
        r_gal : astropy.units.Quantity
            Distance from galaxy's center, i.e. galactic radius.
        unit : str
            Unit of the metallicity. Default is unspecified.
        logOH_solar : float, optional
            Solar metallicity.

        Returns
        -------
        Zprime : number or numpy ndarray
            Predicted metallicity, scaled to solar value
        """

        # making sure units for effective radius are correct
        Re = Re.to(u.radian).value * distance.to(u.kpc)

        # default to a given solar metallicity if not specified 
        if logOH_solar is None:
            logOH_solar = 8.69  # Asplund+09

        # predict gas-phase metallicity for galaxy at 1 Re
        logOH_Re = self.predict_logOH_SAMI19(
            Mstar * 10**0.10)  # Fig. A1 in Sanchez+19
        # extend predictions as a function of radius
        logOH = self.extrapolate_logOH_radially(
            logOH_Re, gradient='CALIFA14',
            Rgal=r_gal, Re=Re)  # Eq. A.3 in Sanchez+14
        
        # assign units if specified (tricky b/c astropy does not
        # have a unit for solar metallicity as of 2026)
        Zprime = (10 ** (logOH - logOH_solar) * u.Unit('')).to(unit)

        self.Zprime=Zprime
        return self.Zprime

    def calc_alpha_co(self, Zprime, sigma_star, Zprime_ll=0.2, Zprime_ul=2.0, 
                        sigma_star_thresh=100, sigma_star_ul=np.inf,
                        J='2-1', method=None):
        """ 
        Calculate the CO to H2 conversion factor (alpha_co) 
        per Table 1 of Leroy and Schinnerer (2024).

        Parameters:
        ----------
        Zprime: number or numpy ndarray
            Predicted metallicity, scaled to solar value
        sigma_star : astropy.units.Quantity
            Stellar mass surface density in solar masses per parsec squared. 
        Zprime_ll: float
            Lower bound on metallicity for prescription to be valid.
        Zprime_ul: float
            Upper bound on metallicity for prescription to be valid.
        sigma_star_thresh: float
            Threshold of stellar mass surface density, above which
            prescription "turns on."
        sigma_star_ul: float
            Upper bound on stellar mass surface density 
            for prescription to be valid.
        J : str
            CO transition.
        method: str
            Method used for calculating mass to light ratio
            (and stellar mass surface density).

        Returns
        -------
        astropy.units.Quantity
            CO to H2 conversion factor (alpha_co).
            Units are solar masses per unit of CO luminosity
            (K km s^-1 pc^2). 

        Notes
        -----
        The result is also stored as an attribute of the object, with name
        depending on the method:

        - alpha_co_gswlc
        - alpha_co_w3w1
        - alpha_co_w4w1
        """

        # establish alpha_co for Milky Way
        alpha_co_mw=4.35 * (u.M_sun/(u.pc)**2) * (1/ ((u.K*u.km)/u.s) )

        # calculate metallicity-dependent term 
        Z_term=np.clip(Zprime, Zprime_ll, Zprime_ul)**(-1.5)

        # calculate starburst emissivity term
        sb_term=(np.clip(sigma_star.value, sigma_star_thresh, sigma_star_ul)/sigma_star_thresh)**(-0.25)

        # add line ratio factor
        if J=='2-1':
            rco=0.65
        if J=='1-0':
            rco=1

        # calculate final alpha_co
        alpha_co=alpha_co_mw * Z_term * sb_term / rco

        attr_name = f'alpha_co_{method}'
        setattr(self, attr_name, alpha_co)
        return getattr(self, attr_name)
    
    def make_alpha_co_map(self, outfile, method=None):

        """ 
        Make a map of the CO to H2 conversion factor.

        Parameters:
        ----------
        outfile: str
            Desired save name of file
        method: str
            Method for calculating mass to light ratio/
            stellar mass surface density/conversion factor. 

        Returns
        -------
        None
            This function produces a FITS file and does not return a value.
        """

        # save header and WCS including correct units
        hdr_new = self.hdr
        out_wcs = WCS(self.hdr)
        hdr_new = out_wcs.to_header()
        hdr_new['BUNIT']='Msun pc-2 (K km s-1)^-1'

        # grab expected size for reshaping to 2D image
        ny=self.hdr['NAXIS1']
        nx=self.hdr['NAXIS1']

        alpha_co = getattr(self, f'alpha_co_{method}', None)

        fits.writeto(outfile, data=self.alpha_co.reshape(ny,nx).value, 
                     header=hdr_new,overwrite=True)
        
    def m_mol(self, distance, method, alpha_co):

        """ 
        Calculate total molecular mass.

        Parameters:
        ----------
        distance: astropy.units.Quantity
            Distance to galaxy.
        method: str
            Method used to derive mass to light ratio/
            stellar surface density/conversion factor.
        alpha_co: astropy.units.Quantity
            CO to H2 conversion factor.

        Returns
        -------
        astropy.units.Quantity
            Total molecular mass (m_mol).
            Units are solar masses. 

        Notes
        -----
        The result is also stored as an attribute of the object, with name
        depending on the method:

        - alpha_co_gswlc
        - alpha_co_w3w1
        - alpha_co_w4w1
        """

        # get the angular pixel scale  
        pix_rad = (self.hdr['CDELT1']*u.deg).to(u.radian)

        # make sure we have units on distance
        if not hasattr(distance, 'unit'):
            raise ValueError("Please give units with the distance.")
        
        # use distance to calculate linear pixel scale
        pix_lin = (pix_rad.value * distance.to(u.pc))**2

        # calculate the CO luminosity (K km s^-1 pc^2)
        l_co = self.data * self.unit * pix_lin
 
        # calculate the total molecular mass 
        m_mol = alpha_co * l_co

        attr_name = f'm_mol_{method}'

        setattr(self, attr_name, m_mol)
        return getattr(self, attr_name)

    # unused functions that may come in handy later

    # def alpha_to_x(self, alpha_co):
    #     x_co = alpha_co.cgs / (2.8*const.m_p.cgs)
    #     x_co = x_co.to(1 / (u.cm**2 * (u.K * u.km / u.s)))

    #     self.x_co=x_co
    #     return self.x_co

    # def calc_nh2(self, x_co, i_co, inc):
    #     i_co = i_co*(u.K*u.km)/u.s # only do this if no units
    #     inc = inc*u.deg
    #     nh2 =  x_co * i_co * np.cos(np.deg2rad(inc))
    #     self.nh2 = nh2
    #     return self.nh2

# defining a function to run methods of Class in one go 
# with the correct input files and parameters, this will
# calculate total molecular mass in a given region.
def calc_m_mol(w1_7p5, w1_15, w3_7p5, w4_15, co,
               gal_sfr, gal_mstar, inc, pa, dist, r_eff):
    
    """ 
    Calculate total molecular mass for an input image.

    Parameters:
    ----------
    wX_Y: str
        Path to input WISE files. 
        7p5 and 15 refer to the resolution in arcseconds. 
    co: str
        Path to input moment zero map of a CO transition.
    gal_sfr: float
        Log of galaxy's star formation rate.
        Assumed units are solar masses per year.
    gal_mstar: float
        Log of galaxy's stellar mass.
        Assumed units are solar masses.
    inc: float
        Galaxy's inclination.
        Assumed units are degrees.
    pa: float
        Galaxy's position angle.
        Assumed units are degrees.
    dist: float
        Distance to galaxy.
        Assumed units are Mpc. 
    r_eff: float
        Galaxy's effective radius.
        Assumed units are arcseconds.

    Returns
    -------
    astropy.table.QTable
        Table with aperture information and
        total molecular mass for each method
        of deriving the mass to light ratio.
    """

    # Store WISE inputs in dictionary for easy retrieval
    wise_inputs = {'w1_7p5': w1_7p5,
                   'w1_15': w1_15,
                   'w3_7p5': w3_7p5,
                   'w4_15': w4_15}
    
    # initialize empty QTable for saving results
    tab=QTable()

    # initialize empty dictionary for saving reprojections
    reprojs={}

    # for the WISE files, make a Map, then reproject the Map
    for label, input in wise_inputs.items():

        map=Map(input)
        reproj=map.reproject(template_file=co)
        reprojs[f'reproj_{label}'] = reproj 

        reproj.add_col(t=tab, name=f'reproj_{label}')

    # make a Map and save as Column to QTable for CO moment 0 file
    co_map=Map(co)
    co_map.add_col(t=tab, name='co_mom0')

    # make a nested dictionary for the different methods of calculating the mass to light ratio 
    methods = {'gswlc': {'map': reprojs['reproj_w1_7p5'],
                         'upsilon_args': {'gal_sfr': gal_sfr, 'gal_Mstar': gal_mstar},
                         'upsilon_attr': 'ups_gswlc',
                         'sig_star_arg': reprojs['reproj_w1_7p5'].data*reprojs['reproj_w1_7p5'].unit,
                         'sig_star_attr': 'sigstar_gswlc'
                         },
               'w3w1': {'map': reprojs['reproj_w1_7p5'],
                         'upsilon_args': {'I_w1': reprojs['reproj_w1_7p5'].data*reprojs['reproj_w1_7p5'].unit, 
                                        'I_w3': reprojs['reproj_w3_7p5'].data*reprojs['reproj_w3_7p5'].unit},
                         'upsilon_attr': 'ups_w3w1',
                         'sig_star_arg': reprojs['reproj_w1_7p5'].data*reprojs['reproj_w1_7p5'].unit,
                         'sig_star_attr': 'sigstar_w3w1'
                         },
               'w4w1': {'map': reprojs['reproj_w1_15'],
                         'upsilon_args': {'I_w1': reprojs['reproj_w1_7p5'].data*reprojs['reproj_w1_7p5'].unit, 
                                        'I_w4': reprojs['reproj_w4_15'].data*reprojs['reproj_w4_15'].unit},
                         'upsilon_attr': 'ups_w4w1',
                         'sig_star_arg': reprojs['reproj_w1_15'].data*reprojs['reproj_w1_15'].unit,
                         'sig_star_attr': 'sigstar_w4w1'
                         }
               }

    # initialize empty dir for storing the M/L ratios and stellar mass surface densities

    for method, input in methods.items():
        ups=input['map'].calc_upsilon(method=method, **input['upsilon_args'])
        # methods[method][f'{method}_gam'] = gam

        sigstar=input['map'].calc_sig_star(upsilon=ups, I_w1=input['sig_star_arg'], i=inc, method=method)
        # methods[method][f'{method}_sigstar'] = sigstar

        # add Column for results of calculating mass to light ratio
        input['map'].add_col(t=tab, name=input['upsilon_attr'], values=ups)
        # add Column for results of calculating stellar mass surface density
        input['map'].add_col(t=tab, name=input['sig_star_attr'], values=sigstar)

    # calculate deprojected radii grid 
    co_map.deproject(center_coord=(co_map.hdr['CRVAL1']*u.deg, co_map.hdr['CRVAL2']*u.deg),
                        incl=inc*u.deg, pa=pa*u.deg, header=co_map.hdr, wcs=WCS(co_map.hdr),
                        naxis=np.array([co_map.hdr['NAXIS1'],co_map.hdr['NAXIS2']]),
                        linear=True,distance=dist*u.Mpc)
    co_map.add_col(t=tab, name='r_G_kpc', values=co_map.radius_kpc)

    # calculate metallicity for deprojected radii grid
    co_map.calc_metallicity(Mstar=gal_mstar, Re=r_eff*u.arcsec, 
                            r_gal=co_map.radius_kpc, distance=dist*u.Mpc)
    co_map.add_col(t=tab, name='Zprime', values=co_map.Zprime)

    # calculate the conversion factor and total molecular mass maps for each method
    for method, input in methods.items():

        sig_star_attr=methods[method]['sig_star_attr']
        sigma_star = getattr(input['map'], sig_star_attr)

        alpha_co=co_map.calc_alpha_co(Zprime=co_map.Zprime,sigma_star=sigma_star)
        co_map.add_col(t=tab, name=f'alpha_co_{method}', values=alpha_co)

        # co_map.make_alpha_co_map(f'/Users/adignan/research/phangs/alpha_co_{method}.fits') # need to require output paths and figure out which files I want 

        m_mol=co_map.m_mol(distance=dist*u.Mpc, alpha_co=alpha_co,method=method)
        co_map.add_col(t=tab, name=f'm_mol_{method}', values=m_mol)

    return tab