# PDF sampler taken from
# https://github.com/nickrodd/NPTFit-Sim/blob/master/NPTFit-Sim/pdf_sampler.py
import numpy as np


class CDFSampler:
    def __init__(self, xvals, cdf):
        """ At outset sort and calculate CDF so not redone at each call
        :param xvals: array of x values
        :param cdf: cdf function that can be used instead, so there is no need to sample a new cdf
        """
        self.cdf = cdf
        # print("START")
        # print("cdf")
        # print(self.cdf)
        # print(self.cdf-1)
        self.xvals = xvals
        # print("xvals")
        # print(self.xvals)
        self.sortxvals = np.argsort(self.cdf) #probably obsolete



        # new_cdf=np.zeros(shape=cdf.size-1)
        # for cdf_index in range(0,cdf.size-1):
        #     new_cdf[cdf_index]=cdf[cdf_index+1]-cdf[cdf_index] #calculate probability mass for a photon to land in a bin


    def __call__(self, samples):
        """
        When class called returns samples number of draws from pdf
        :param samples: number of draws you want from the pdf
        :returns: number of random draws from the provided PDF
        """
        # Random draw from a uniform, up to max of the cdf, which need
        # not be 1 as the pdf does not have to be normalised
        #print("Psamples shape: " + str(samples.shape) + "samples: " + str(samples[:10]))

        unidraw = np.random.uniform(low=self.cdf[0], high=self.cdf[-1], size=samples) #Pfusch low ist nicht mehr 0.0 (default) sondern lowest cdf nr
        cdfdraw = np.searchsorted(self.cdf, unidraw) #gibt indizes von cdf aus wo unidraw werte sind
        # print("unidraw")
        # print(unidraw)
        # print(unidraw-1)
        # print("CDFDRAW")
        # print(cdfdraw)
        # print(cdfdraw-1)
        # print("sortxvals")
        # print(self.sortxvals)
        # print(self.sortxvals[cdfdraw])
        cdfdraw = self.sortxvals[cdfdraw] #sucht index xwerte von den gezogenen cdf values
        return self.xvals[cdfdraw]  #xwerte für die dazugehörigen cdf werte #mayb obsolete


