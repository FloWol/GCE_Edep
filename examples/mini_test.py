import ray
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os
import sys

sys.path.append("/home/flo/GCE_NN/")
loop=int(sys.argv[1])
print(str(loop) + "Anfang")

import GCE.gce
gce = GCE.gce.Analysis()
gce.load_params("../parameter_files/parameters.py")
#gce.print_params()
#gce.psf_make_map("bub")

# Ray settings (for parallelized data generation)
# ray.shutdown() #delete old rays if there are any
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
ray_settings = {"num_cpus": 4}  # select the number of CPUs here
gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=loop)

gce.combine_template_maps(save_filenames=True, do_combine=True)
