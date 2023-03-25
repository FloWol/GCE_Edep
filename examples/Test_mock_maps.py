import ray
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import healpy as hp
import os

import GCE.gce
gce = GCE.gce.Analysis()
gce.load_params("../parameter_files/parameters.py")





gce.build_pipeline()
n_samples = 50
test_samples = gce.datasets["train"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][0]
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%

#gce.plot_mean_spectra(test_data)
#gce.plot_mean_spectra_template(test_data)
#gce.plot_spectra(test_data)

gce.build_nn()

#gce.train_nn("flux_fractions")
gce.load_nn()


pred = gce.predict(test_data, tau=tau, multiple_taus=True)  # get the NN predictions
# gce.plot_flux_ebins_with_color_flux(test_data,test_ffs, pred)
# gce.plot_flux_fractions_Ebin(test_ffs, pred)
# gce.plot_flux_fractions_total(test_ffs, pred)
#
# gce.plot_templates_scaled_ff(test_ffs, pred)

#gce.plot_ebin_ff(test_ffs, pred)


gce.plot_flux_fractions_fermi(pred,test_data,  Flux=True, Esquared=True)
