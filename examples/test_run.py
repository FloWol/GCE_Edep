import ray
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os

import GCE.gce
gce = GCE.gce.Analysis()
gce.load_params("../parameter_files/parameters.py")
gce.print_params()

# Ray settings (for parallelized data generation)
ray.shutdown() #delete old rays if there are any
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
ray_settings = {"num_cpus": 4}  # select the number of CPUs here
gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=0)

gce.combine_template_maps(save_filenames=True, do_combine=True)

gce.build_pipeline()

samples = gce.datasets["test"].get_samples(1)
data, labels = samples["data"], samples["label"]  # samples contains data and labels (flux fractions & SCD histograms)
print("Shapes:")
print("  Flux fractions", labels[0].shape)  # n_samples x n_templates

gce.build_nn()

#gce.train_nn("flux_fractions")
gce.load_nn()

n_samples = 20
test_samples = gce.datasets["test"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][0]
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
pred = gce.predict(test_data, tau=tau, multiple_taus=True)  # get the NN predictions

gce.plot_flux_fractions_Ebin(test_ffs, pred)
gce.plot_flux_fractions_total(test_ffs, pred)
gce.plot_flux_per_Ebin(test_ffs, pred)

