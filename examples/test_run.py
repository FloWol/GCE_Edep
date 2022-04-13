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

#plot generated maps
map_to_plot = 0 #index of which map is plotted
r = gce.p.data["outer_rad"] + 1
fig, ax= plt.subplots(2,7,figsize = (25,5))
subplot=1
[axi.set_axis_off() for axi in ax.ravel()]
plt.tight_layout()
for Ebin in range(0,len(gce.p.data["Ebins"])-1):

    hp.cartview(gce.decompress(data[map_to_plot,:,Ebin] * gce.template_dict["rescale_compressed"]), nest=True,
                 lonra=[-r, r], latra=[-r, r], sub=(2,7,subplot), title='Counts Bin ' + str(Ebin))

    hp.cartview(gce.decompress(data[map_to_plot,:,Ebin]), nest=True,
                 lonra=[-r, r], latra=[-r, r], sub=(2,7,7+subplot), title='Flux Bin ' + str(Ebin))
    subplot+=1

plt.show()


hp.cartview(gce.decompress(gce.template_dict["rescale_compressed"], fill_value=np.nan), nest=True,
                title="Fermi exposure correction", lonra=[-r, r], latra=[-r, r])
plt.show()
fermi_counts = gce.datasets["test"].get_fermi_counts()
hp.cartview(gce.decompress(fermi_counts * gce.generators["test"].settings_dict["rescale_compressed"]), nest=True,
            title="Fermi data: Count space", max=100, lonra=[-r, r], latra=[-r, r])
# hp.cartview(gce.decompress(fermi_counts), nest=True, title="Fermi data: Flux space", max=100)
plt.show()


gce.build_nn()

gce.train_nn("flux_fractions")



