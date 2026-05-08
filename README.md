# co_to_h2
Python code to implement various mass to light ratio and stellar mass surface density prescriptions to (1) calculate the CO-to-H2 conversion factor to (2) the total molecular mass in a given region from radio observations of CO (2–1).  

# Table of contents
[`co_to_h2.py`](https://github.com/annadignan/co_to_h2/blob/main/co_to_h2.py): This module performs the main calculations and includes helper functions to plot the results and save as .csv files.
* `plotmap`: helper function for plotting
* `photometry`: helper function for performing aperture photometry
* `Map`: class to load in a FITS file or 2D array for manipulation.
  * `reproject`: reproject a FITS file onto a template grid and WCS.
  * `add_col`: flatten a 2D array and store as a column in a table.
  * `calc_upsilon`: calculate the mass to light ratio using either the GSWLC specific star formation rate, WISE3-to-WISE1 color, or WISE4-to-WISE1 color predictions from [Leroy (2019)](https://iopscience.iop.org/article/10.3847/1538-4365/ab3925).
  * `calc_sig_star`: calculate the stellar mass surface density based on the calibration from [Leroy (2021)](https://iopscience.iop.org/article/10.3847/1538-4365/ac17f3) for WISE1 intensity maps.
  * `deproject`: calculate deprojected radii and projected angles for a galaxy disk
  * `predict_logOH_SAMI19`: calculate the gas phase abundance from mass-metallicity relations per [Sánchez (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.3042S/abstract).
  * `extrapolate_logOH_radially`: extrapolate the abundance within a galaxy assuming a radial gradient per [Sánchez (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...563A..49S/abstract).
  * `calc_metallicity`: calculate the metallicity based on the predicted gas phase abundance.
  * `calc_alpha_co`: calculate the CO-to-H2 conversion factor per [Schinnerer and Leroy (2024)](https://arxiv.org/abs/2403.19843).
  * `make_alpha_co_map`: save the output from `calc_alpha_co` as a FITS file.
  * `m_mol`: calculate the total molecular mass based on the CO-to-H2 conversion factor and a given CO moment 0 map.
* `calc_m_mol`: extend the calculation of total molecular mass to multiple galaxies and/or regions.

# Installation
After downloading co_to_h2.py, you can use the code in a Jupyter notebook by running the following in a cell: 

```
%load_ext autoreload
%autoreload 2

import co_to_h2
```

# Dependencies
* `numpy`
* `matplotlib`
* [`astropy`](https://docs.astropy.org/en/stable/install.html#installing-astropy)
* [`photutils`](https://photutils.readthedocs.io/en/stable/)
* [`reproject`](https://reproject.readthedocs.io/en/stable/)
* [`CO_conversion_factor`](https://github.com/astrojysun/COConversionFactor)

# Credits
If any tools in this repository are used in a publication, please cite the relevant source papers linked above. Also consider giving credit to Adam Leroy and Jiayi Sun for their Python implementations used in this work.  
