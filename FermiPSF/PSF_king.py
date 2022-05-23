################################################################################
# PSF_king.py
################################################################################
# Written: Nick Rodd (CERN) - 17 May 2022
################################################################################
#
# Determine the king function parameters needed to mode the Fermi PSF at an
# arbitrary energy
# 
# For details see:
# fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html
#
# NB: At present, results will only be produced for Pass8 SOURCE and
# ULTRACLEANVETO data, other datasets will require adding additional psf files
#
################################################################################

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d


def params(eventclass, quartile, energy):
    """
    Class for determining the king function PSF parameters

    Inputs:
        eventclass: data class, 2 = SOURCE or 5 = ULTRACLEANVETO
        quartile: PSF quartile, 1-4
        energy: energy value to evaluate the PSF at [GeV]

    Outputs:
        Return an array of the six relevant King PSF parameters: fcore, score, 
        gcore, stail, gtail, SpE

    """

    ### Load the PSF file
    if eventclass == 2:
        psf_file_name = 'psf_P8R2_SOURCE_V6_PSF.fits'
    elif eventclass==5:
        psf_file_name = 'psf_P8R2_ULTRACLEANVETO_V6_PSF.fits'

    f = fits.open('/home/flo/GCE_NN/FermiPSF/psf_data/' + psf_file_name)

    ### Establish the auxiliary parameters
    if quartile == 1:
        params_index = params_index_Q1
        rescale_index = rescale_index_Q1
        if eventclass == 2: theta_norm = theta_norm_Q1_EV2
        if eventclass == 5: theta_norm = theta_norm_Q1_EV5

    if quartile == 2:
        params_index = params_index_Q2
        rescale_index = rescale_index_Q2
        if eventclass == 2: theta_norm = theta_norm_Q2_EV2
        if eventclass == 5: theta_norm = theta_norm_Q2_EV5

    if quartile == 3:
        params_index = params_index_Q3
        rescale_index = rescale_index_Q3
        if eventclass == 2: theta_norm = theta_norm_Q3_EV2
        if eventclass == 5: theta_norm = theta_norm_Q3_EV5

    if quartile == 4:
        params_index = params_index_Q4
        rescale_index = rescale_index_Q4
        if eventclass == 2: theta_norm = theta_norm_Q4_EV2
        if eventclass == 5: theta_norm = theta_norm_Q4_EV5

    ### Load the appropriate data and preprocess
    rescale_array = f[rescale_index].data[0][0]
    
    E_min = f[params_index].data[0][0] # size is 23
    E_max = f[params_index].data[0][1]
    theta_min = f[params_index].data[0][2] # size is 8
    theta_max = f[params_index].data[0][3]
    NCORE = np.array(f[params_index].data[0][4]) # shape is (8, 23)
    NTAIL = np.array(f[params_index].data[0][5])
    SCORE = np.array(f[params_index].data[0][6])
    GCORE = np.array(f[params_index].data[0][8])
    STAIL = np.array(f[params_index].data[0][7])
    GTAIL = np.array(f[params_index].data[0][9])
    FCORE = np.array([[1/(1+NTAIL[i,j]*STAIL[i,j]**2/SCORE[i,j]**2) 
                         for j in range(np.shape(NCORE)[1])] 
                             for i in range(np.shape(NCORE)[0])])

    theta_norm = np.transpose([theta_norm for i in range(23)])

    ### Interpolate to obtain all the values
    FCORE_int = interp1d((E_max+E_min)/2.*10**-3, np.sum(theta_norm*FCORE,axis=0))
    ofcore = FCORE_int(energy)
    SCORE_int = interp1d((E_max+E_min)/2.*10**-3, np.sum(theta_norm*SCORE,axis=0))
    oscore = SCORE_int(energy)
    GCORE_int = interp1d((E_max+E_min)/2.*10**-3, np.sum(theta_norm*GCORE,axis=0))
    ogcore = GCORE_int(energy)
    STAIL_int = interp1d((E_max+E_min)/2.*10**-3, np.sum(theta_norm*STAIL,axis=0))
    ostail = STAIL_int(energy)
    GTAIL_int = interp1d((E_max+E_min)/2.*10**-3, np.sum(theta_norm*GTAIL,axis=0))
    ogtail = GTAIL_int(energy)

    oSpE = np.sqrt((rescale_array[0]*(energy*10**3/100)**(rescale_array[2]))**2 + rescale_array[1]**2)

    ### Return all parameters in a single array
    return np.array([ofcore, oscore, ogcore, ostail, ogtail, oSpE])
        

####################
# Auxiliary Params #
####################

# Define the auxiliary parameters needed to construct the PSF

### Parameter Index
# Index denoting where the appropraite parameters are stored in the PSF file
params_index_Q1 = 10
params_index_Q2 = 7
params_index_Q3 = 4
params_index_Q4 = 1

### Rescale Index
# Index for the energy rescaling factor that enters SpE
rescale_index_Q1 = 11
rescale_index_Q2 = 8
rescale_index_Q3 = 5
rescale_index_Q4 = 2

### Theta Norm
# Weighting factor for different photon incident angles
theta_norm_Q1_EV2 = [0.0000000,9.7381019e-06,0.0024811595,0.022328802,
                     0.080147663,0.17148392,0.30634315,0.41720551]
theta_norm_Q1_EV5 = [0.0000000,9.5028121e-07,0.00094418357,0.015514370,
                     0.069725775,0.16437751,0.30868705,0.44075016]

theta_norm_Q2_EV2 = [0.0000000,0.00013001938,0.010239333,0.048691643,
                     0.10790632,0.18585539,0.29140913,0.35576811]
theta_norm_Q2_EV5 = [0.0000000,1.6070284e-05,0.0048551576,0.035358049,
                     0.091767466,0.17568974,0.29916159,0.39315185]

theta_norm_Q3_EV2 = [0.0000000,0.00074299273,0.018672204,0.062317201,
                     0.12894928,0.20150553,0.28339386,0.30441893]
theta_norm_Q3_EV5 = [0.0000000,0.00015569366,0.010164870,0.048955837,
                     0.11750811,0.19840060,0.29488095,0.32993394]

theta_norm_Q4_EV2 = [4.8923139e-07,0.011167475,0.092594658,0.15382001,
                     0.16862869,0.17309118,0.19837774,0.20231968]
theta_norm_Q4_EV5 = [0.0000000,0.0036816313,0.062240006,0.14027030,
                     0.17077023,0.18329804,0.21722594,0.22251374]
