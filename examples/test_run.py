import ray
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import healpy as hp
import os

import GCE.gce
gce = GCE.gce.Analysis()
gce.load_params("../parameter_files/parameters.py")
#gce.print_params()
#gce.psf_make_map("gce_12_PS")

# Ray settings (for parallelized data generation)
# ray.shutdown() #delete old rays if there are any
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
# ray_settings = {"num_cpus": 4}  # select the number of CPUs here
# gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=0)
# #
# gce.combine_template_maps(save_filenames=True, do_combine=True)
#
gce.build_pipeline()
#
#
gce.build_nn()
# #
# gce.train_nn("flux_fractions")
gce.load_nn()

n_samples = 10
test_samples = gce.datasets["train"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][0]
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
pred = gce.predict(test_data, tau=tau, multiple_taus=True)  # get the NN predictions


#abv_thresh, outlier_pred, outlier_errors, outlier_true = gce.plot_outliers(test_ffs, pred, threshold=0.11, only_errors=False,show_mapID=False)
#gce.plot_flux_fractions_Ebin(test_ffs, pred)
gce.plot_flux_fractions_total(test_ffs, pred)
#
#
#

gce.plot_templates_scaled_ff(test_ffs, pred)

# gce.plot_flux_per_Ebin(test_data, test_ffs, pred, 18)
# gce.plot_ff_per_Ebin(test_ffs, pred, 18)

# for image in abv_thresh[0]: #np.unique(abv_thresh)[0]
#     title = "Map Nr. " + str(image)
#     gce.plot_flux_per_Ebin(test_data, test_ffs, pred, image, title=title)
#
#     #expand dims to make it compatible with folowing functions
#     y =pred["ff_mean"][image]
#     y =tf.expand_dims(
#         y, 0, name=None
#     )
#
#     ye=pred["ff_logvar"][image]
#     ye = tf.expand_dims(
#         ye, 0, name=None
#     )
#     x = np.expand_dims(test_ffs[image], 0) # original value at imageth place dim 1,pix,ebin
#
#     help_dict={"ff_mean": y, "ff_logvar": ye}
#     #gce.plot_ff_per_Ebin(test_ffs, pred, image)
#     #gce.plot_flux_fractions_Ebin(x, help_dict)



