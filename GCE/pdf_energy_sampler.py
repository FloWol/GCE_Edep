# PDF sampler taken from
# https://github.com/nickrodd/NPTFit-Sim/blob/master/NPTFit-Sim/pdf_sampler.py
import numpy as np


class PDFSampler:
    def __init__(self, xvals, pofx_list, Ebins):
        """ At outset sort and calculate CDF so not redone at each call
        :param xvals: array of x values
        :param pofx: array of associated p(x) values (does not need to be
               normalised)
        """
        self.n_Ebins = Ebins.size
        self.xvals = xvals
        self.pofx_list = pofx_list

        self.cdf_list = []
        self.sortxvals_list=[]

        for pofx in pofx_list:
            # Check p(x) >= 0 for all x, otherwise stop
            assert(np.all(pofx >= 0)), "pdf cannot be negative"

            # Sort values by their p(x) value, for more accurate sampling
            self.sortxvals = np.argsort(pofx) # pofx indizes die pofx aufsteigend sortieren
            self.sortxvals_list.append(self.sortxvals)
            self.pofx = pofx[self.sortxvals] #pofx wird aufsteigend sortiert

            # Calculate cdf
            self.cdf = np.cumsum(self.pofx)
            self.cdf_list.append(self.cdf)


    def __call__(self, Eind):
        """
        When class called returns samples number of draws from pdf
        :param samples: number of draws you want from the pdf
        :returns: number of random draws from the provided PDF
        """
        # Random draw from a uniform, up to max of the cdf, which need
        # not be 1 as the pdf does not have to be normalised
        #print("Psamples shape: " + str(samples.shape) + "samples: " + str(samples[:10]))

        assert min(Eind) == max(Eind), "passed energy values are not of one number only"
        assert Eind.size != 0, 'empty array'

        # distances=np.zeros(shape=Eind.shape)
        #
        # #old version not needed for energy dependent templates
        # for index in range(0, self.n_Ebins-1):
        #     Ebin_indices = np.argwhere(Eind==index).flatten() #save all indices of photons within the same energy bin
        #
        #     unidraw = np.random.uniform(high=self.cdf_list[index][-1], size=Ebin_indices.size) #self.cdf_list[index][-1] = highest value of cdf of given Ebin
        #     cdfdraw = np.searchsorted(self.cdf_list[index], unidraw) #gibt indizes von cdf aus wo unidraw werte sind
        #     cdfdraw = self.sortxvals_list[index][cdfdraw] #sucht index xwerte von den gezogenen cdf values
        #     distances[Ebin_indices] = self.xvals[cdfdraw] #fügt allen photonen eines energie bins ihre x werte zur cdf
        #     #xvals_list.append(self.xvals[cdfdraw])

        index=Eind[0]

        unidraw = np.random.uniform(high=self.cdf_list[index][-1], size=Eind.size)
        cdfdraw = np.searchsorted(self.cdf_list[index], unidraw)  # gibt indizes von cdf aus wo unidraw werte sind
        cdfdraw = self.sortxvals_list[index][cdfdraw]  # sucht index xwerte von den gezogenen cdf values
        distances= self.xvals[cdfdraw]  # fügt allen photonen eines energie bins ihre x werte zur cdf
        # xvals_list.append(self.xvals[cdfdraw])
        return distances

        #noch genauer versuchen den Sampler zu verstehen




        #lösungen:
        # loop über Eind und immer checken
        # oder indizes checken und sortieren

        # arghwere for loop über alle energy bins
        # für jede liste dann einen unidraw cdfdraw etc und das ganze dann wieder in eine liste returnen die
        # samples ändert sich und self.cdf muss aus cdf_list kommen
        # noch genau anschauen wie das mit self.xvals und self.sortxvals sunst


