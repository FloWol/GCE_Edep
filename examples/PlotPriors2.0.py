import pickle
import sys
import ray
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import healpy as hp
import os

import GCE.gce
gce = GCE.gce.Analysis()
gce.load_params("../parameter_files/parameters.py")

fermi_data=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/fermidata_counts.npy")
fermi_exposure=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/fermidata_exposure.npy")
fermi_mask=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/fermidata_pscmask_3fgl.npy")

fermi_bub=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/template_bub_smooth.npy")
fermi_gce=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/template_nfw_g1p2.npy")
fermi_iso=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/template_iso_smooth.npy")
fermi_pibs=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/template_Opi.npy")
fermi_ic=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/template_Oic.npy")
fermi_disk=np.load("/home/flo/GCE_NN/data/fermi_data_edep/fermi_data_256/template_dsk_z0p3.npy")



templates=[fermi_pibs,fermi_ic,fermi_iso,fermi_bub,fermi_gce,fermi_disk]
names=["Pi+Brems", "IC", "ISO", "Fermi Bubbles", "GCE", "disk"]







def plot_temps(moll=True):
    n_row = 6
    n_col = 10
    subplot = 1
    r = 26
    if moll==True:
        for row in range(0,n_row):
            for ebin in range(0,n_col):
                if row < 2:
                    #hp.mollview(templates[row][ebin],sub=(6,10,subplot))
                    hp.mollview(templates[row][ebin], title=names[row])
                else:
                    hp.mollview(templates[row][10+ebin], title=names[row])
                plt.savefig("moll "+str(names[row])+" Ebin "+str(ebin)+".png")
            plt.show()

    else:
        for row in range(0,n_row):
            for ebin in range(0,n_col):
                if row < 2:
                    #hp.mollview(templates[row][ebin],sub=(6,10,subplot))
                    hp.cartview(templates[row][ebin],
                                lonra=[-r, r], latra=[-r, r], title=names[row])
                else:
                    hp.cartview(templates[row][10+ebin],
                                lonra=[-r, r], latra=[-r, r], title=names[row])
                plt.savefig("cart "+str(names[row])+" Ebin "+str(ebin)+".png")
            plt.show()

def plot_masks(moll=True):
    r = 26
    n_col = 10
    if moll==True:
        for ebin in range(0,n_col):
            hp.mollview(fermi_mask[10+ebin], title="Moll Mask Ebin "+str(ebin))
            plt.savefig("moll Mask Ebin "+str(ebin)+".png")
            plt.show()

    else:
        for ebin in range(0,n_col):
            hp.cartview(fermi_mask[10+ebin], title="Cart Mask Ebin "+str(ebin))
            plt.savefig("Cart Mask Ebin "+str(ebin)+".png")

            plt.show()
def plot_exposure(moll=True):
    r = 26
    n_col = 10
    if moll==True:
        for ebin in range(0,n_col):
            hp.mollview(fermi_exposure[10+ebin], title="Moll Exp Ebin "+str(ebin))
            plt.savefig("moll Exp Ebin "+str(ebin)+".png")
            plt.show()

    else:
        for ebin in range(0,n_col):
            hp.cartview(fermi_exposure[10+ebin], title="Cart Exp Ebin "+str(ebin))
            plt.savefig("Cart Exp Ebin "+str(ebin)+".png")

            plt.show()


plot_temps(moll=False)
plot_masks(moll=False)
plot_exposure(moll=False)
plot_temps()
plot_masks()
plot_exposure()
print("done")
#Ebins kontorllieren
#TODO PSF, I/O





