import scipy.optimize as opt
from base_optimiser import AbstractOptimiser


class FminOptimiser(AbstractOptimiser):
    # choose one of optimisers in scipy
    def optimise_fmin(self, initial_parameters):
        opt.fmin(self.evaluate(), initial_parameters)

    def optimise_evolution(self, bounds):
        # Differencial Evolution Method
        opt.differential_evolution(self.evaluate(), bounds, strategy='best1bin',
                                   maxiter=1000, popsize=20, tol=0.01,
                                   mutation=(0.5, 0.8), recombination=0.2)
