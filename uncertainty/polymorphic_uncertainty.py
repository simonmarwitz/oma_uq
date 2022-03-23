import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats.qmc
import scipy.optimize
from itertools import product, chain
from uncertainty.data_manager import HiddenPrints, simplePbar

class UncertainVariable(object):
    '''
    Implementation of various types of uncertain variables
    to be used with uniform quasi-random sampling over the support
    that is why they can not be directly sampled
    '''
    def __init__(self, name, children, primary=True):
        self.value = None
        self.name = name
        self.primary = primary
        
        for child in children:
            if isinstance(child, UncertainVariable):
                is_poly = True
                break
        else:
            is_poly = False
        
        self.is_poly = is_poly
        
    def support(self, ):
        '''
        getting the support is usually needed in pre-processing, before propagation
        no frezzing is necessary, simply evaluating the minimal and maximum values possible
        '''
        raise NotImplementedError("When subclassing UncertainVariable you should implement a support() method.")
            
    def freeze(self, value):
        # fix variable, make it a "CertainVariable"
        assert isinstance(value, (int,float))
        assert not np.isnan(value)
        self.value = value
    
    def unfreeze(self, ):
        self.freeze(None)
    
    @property
    def _children(self):
        raise NotImplementedError("When subclassing UncertainVariable you should implement a _children property.")
    
    @property
    def frozen(self, ):
        return self.value is not None
    
    def __repr__(self,):
        return self.name
    
class RandomVariable(UncertainVariable):
    
    def __init__(self, dist, name, params, primary=True):
        super().__init__(name, params, primary)
        
        self.dist = dist
        self.dist_fun = getattr(scipy.stats, self.dist)
        self.params = params
    
    @property
    def _children(self):
        if self.is_poly:
            return self.params
        else:
            return ()
    
    def support(self, percentiles=(0.001,0.999)):
        eval_params = []
        for param in self.params:
            if isinstance(param, UncertainVariable):
                param = param.support(percentiles=percentiles)
            if isinstance(param, (float, int)):
                param = [param]
            eval_params.append(param)
        supp = [np.infty, -np.infty]
        for params in product(*eval_params): # nested for loop over all eval_params -> cartesian product = full factorial
            rv = self.dist_fun(*params)
            this_supp = rv.ppf(percentiles)
            supp[0] = min(supp[0], this_supp[0])
            supp[1] = max(supp[1], this_supp[1])
        if np.isnan(supp).any():
            raise RuntimeError(f"Variable {self} has invalid support. Check your definitions")
        return supp
    
    def rvs(self, size=1):
        # we need to have all uncertain parameters fixed
        eval_params = []
        for param in self.params:
            if isinstance(param, UncertainVariable):
                if not param.frozen:
                    raise RuntimeError(f"Variable {param.name} is not frozen.")
                eval_params.append(param.value)
            else:
                eval_params.append(param)
        rv = self.dist_fun(*eval_params)
        
        return rv.rvs(size=size)
    
    def prob_dens(self, values=None):
        # we need to have all uncertain parameters fixed
        eval_params = []
        for param in self.params:
            if isinstance(param, UncertainVariable):
                if not param.frozen:
                    raise RuntimeError(f"Variable {param.name} is not frozen.")
                eval_params.append(param.value)
            else:
                eval_params.append(param)
        # rv = self.dist_fun(*eval_params)
        # pdf = rv.pdf(*eval_params,values)
        pdf = self.dist_fun.pdf(values, *eval_params)
        return pdf
#         return pdf / pdf.sum()
    
    def __repr__(self,):
        return f'<RV: {self.name}>'
    
class MassFunction(UncertainVariable):
    # currently only continuous-valued belief functions are supported
    # may be extended to set-valued belief functions 
    # Another Class fur fuzzy sets may be created, exposing the same set of methods
    '''
       - properties such as frame, core, bel, pl, q, focal, singletons, 
   - transformations: pignistic, normalize, inverse pignistic (defining a mass function from samples)
   
   incremental focal sets should be implemented, this is important for polymorphy only
       implement internal storage of incremental focals sets
           return non-incremental focal sets
           
    '''
    
    def __init__(self, name, focals, masses, frame=None, primary=True, incremental=False):
        super().__init__(name, chain.from_iterable(focals), primary)
        
        # focals may be a 3-tuple, where the first is the incremental reference and will be dropped in numeric_focal,support etc
        # focals may be a 1-tuple to define a crisp focal set, containing only singletons
        # otherwise it should be a 2-tuple
        
        if not isinstance(focals, np.ndarray): # note, it may contain objects
            focals_l = focals
            focals = np.empty((len(focals),3), dtype=object)
            for focal_arr,focal in zip(focals,focals_l):
                if len(focal) == 1:
                    focal_arr[:] = np.nan
                    focal_arr[1] = focal[0]
                elif len(focal) == 2:
                    focal_arr[1:] = focal
                    focal_arr[0] = np.nan
                    if not (isinstance(focal[0], UncertainVariable) or isinstance(focal[1], UncertainVariable)):
                        assert focal[1] >= focal[0]
                else:
                    focal_arr[:] = focal
                    incremental = True
            
        if not isinstance(masses,  np.ndarray):
            masses = np.array(masses)

        assert focals.shape[0] == masses.shape[0]
        
        total_mass = np.sum(masses)
        if total_mass > 1:
            raise ValueError(f'The sum of all mass values ({total_mass}) must not be greater than one.')
        if  total_mass < 1:
            if frame is None:
                raise ValueError("If unassigned mass remains, the frame must be specified")
            focals = np.vstack([focals, frame])
            masses = np.vstack([masses, [1 - total_mass]])
        
        self.incremental = incremental
        self._focals = focals
        self.masses = masses
        self.frame = frame
    
    @property
    def n_focals(self,):
        return self._focals.shape[0]
    
    @property
    def numeric_focals(self,):
        # evaluates all focal sets to numeric values
        # raises error if nested UncertaunVariables are not frozen
        
        incremental = self.incremental
        focals = self._focals
        numeric_focals = np.empty((self.n_focals, 2))
        for i, (incvar, lbound, ubound) in enumerate(focals): # iterates over rows
            
#             print(i, (incvar, lbound, ubound))
            if isinstance(lbound, UncertainVariable):
                if not lbound.frozen:
                    raise RuntimeError(f"Variable {lbound.name} is not frozen.")
                lbound = lbound.value
                            
            if isinstance(ubound, UncertainVariable):
                if not ubound.frozen:
                    raise RuntimeError(f"Variable {ubound.name} is not frozen.")
                ubound = ubound.value
            elif np.isnan(ubound): # crisp set / interval / singleton
                if incremental: ubound = 0
                else: ubound = lbound
                
            if isinstance(incvar, UncertainVariable):
                if not incvar.frozen:
                    raise RuntimeError(f"Variable {incvar.name} is not frozen.")
                incvar = incvar.value
    
#             print(i, numeric_focals.shape, lbound, ubound, incvar)
            if incremental and np.isnan(incvar):
                numeric_focals[i, :] = (lbound, lbound + ubound)
            elif incremental:
                numeric_focals[i, :] = (incvar + lbound, incvar + ubound)
            else:    
                numeric_focals[i, :] = (lbound, ubound)
                
        return numeric_focals
    
    @property
    def _children(self):
        if self.is_poly:
            return self._focals.ravel()
        else:
            return ()
        
    def support(self, *args, **kwargs):
        incremental = self.incremental
        focals = self._focals
        supp = [np.infty, -np.infty]
        for incvar, lbound, ubound in focals: # iterates over rows
            
            if isinstance(lbound, UncertainVariable):
                lsupp = lbound.support(*args, **kwargs) 
            else:
                lsupp = (lbound, lbound)
                
            if isinstance(ubound, UncertainVariable):
                ubound = ubound.support(*args, **kwargs)[1]
            elif np.isnan(ubound):  # crisp set / interval / singleton
                ubound = 0
                
            if isinstance(incvar, UncertainVariable):
                incsupp = incvar.support(*args, **kwargs)
            else:
                incsupp = (incvar, incvar)
    
            if incremental and isinstance(incvar, UncertainVariable):
                lbound = incsupp[0] + lsupp[0]
                ubound += incsupp[1]
            elif incremental:
                lbound = lsupp[0]
                ubound += lsupp[1]
            else:    
                lbound = lsupp[0]
            
            supp[0] = min(supp[0], lbound)
            supp[1] = max(supp[1], ubound)

        if np.isnan(supp).any():
            raise RuntimeError(f"Variable {self} has invalid support. Check your definitions")
        return supp
    
    def mass(self, value=None):
        # only single values may be provided        
        # value may be in more than one focal set, bel and pl are better used instead
        # may return multiple or no mass values
        numeric_focals = self.numeric_focals
        
        masses = self.masses
        inds = np.logical_and(numeric_focals[:,0] >=value,
                              numeric_focals[:,1] <=value)
        return masses[inds]
    
    def __repr__(self,):
        return f'<MF: {self.name}>'



    
def approximate_out_intervals(output, hyc_dat_inds):
    n_hyc = hyc_dat_inds.shape[0]
    # for each input hypercube get the output focal set of the output
    intervals = np.full((n_hyc, 2), np.nan)
    for i_hyc in range(n_hyc):
        selector = hyc_dat_inds[i_hyc, :]
        if selector.any():
            this_output = output[selector]
            intervals[i_hyc, :] = np.min(this_output), np.max(this_output)
        else:
            pass
            #print('Encountered empty hypercube -> no focal set assigned.')
    return intervals 

def optimize_out_intervals(mapping, arg_vars, hyc_foc_inds, vars_epi, vars_ale=None, fun_kwargs=None):
    
    print('This function is not implemented correctly, currently a placeholder, a wrapper must be written to account for arg_vars')
    '''
    This function may either be used to quantify incompleteness or imprecision:
        incompleteness: 
            mapping is stat_fun, needs stochastic samples and weights, weights are obtained by freezing aleatory variables in each optimization step (need a wrapper around stat_fun)
            arg_vars is a dict mapping argument names to stochastic samples and weights
            hyc_foc_inds are inc_hyc_foc_inds
            vars_epi are vars_inc
            vars_ale are not needed?
            fun_kwargs are additional function keyword arguments, e.g. histogram bins, etc.
        imprecision:
            mapping is model function
            arg_vars is a dict mapping argument names to variable names
            hyc_foc_inds are imp_hyc_foc_inds
            vars_epi are vars_imp
            vars_ale are not needed?
            fun_kwargs are additional function keyword arguments, e.g. model parameters
    bounds and initial conditions for each hypercube are obtained from vars_epi for each variable in arg_vars
    
    
    '''
    
#     this_out = mapping(**{arg:samples[var].iloc[ind_ale] for arg, var in arg_vars.items() if var in names_ale}, 
#                    **{arg:samples[var].iloc[ind_epi] for arg, var in arg_vars.items() if var in names_epi},)
    numeric_focals = [var.numeric_focals for var in vars_epi]
    n_hyc = len(hyc_foc_inds)
    intervals = np.full((n_hyc ,2), np.nan)
    
    for i_hyc in range(n_hyc):
        
        bounds = [...]
        init = [np.mean(bound) for bound in bounds]
        
        resl = scipy.optimize.minimize(lambda x, args: mapping(*x, args), init, fun_args, bounds = bounds)
        resu = scipy.optimize.minimize(lambda x, args: -mapping(*x, args), init, fun_args, bounds = bounds)
        
        intervals[i_hyc, :] = [resl.fun, -resu.fun]
    
    return intervals

def compute_belief(focals, masses, cumulative=False, bins=None):
    if bins is None:
        bins = np.sort(focals.ravel())
    elif not isinstance(bins, (np.ndarray, list, tuple)):
        # assuming bins = nbins -> int: number of bins
        bins = np.linspace(focals.min(), focals.max(), int(bins))

    bel = np.zeros(bins.size)
    pl = np.zeros(bins.size)
    q = np.zeros(bins.size)

    # Bins are sets A
    # Focals are sets B or C
    for i in range(bins.size - 1):
        # set A
        lbin, ubin = bins[i:i+2]
        if cumulative:
            lbin = focals.min()

        # get all sets B that are a subset of A
        # that means the lower boundary of B must be greater or equal than the lower boundary of A
        # and similarly for the upper boundary
        belinds = np.logical_and(focals[:,0] >= lbin,
                                 focals[:,1] <= ubin)
        bel[i] = np.sum(masses[belinds])
        # get all sets B that intersect with A
        # that means the lower boundary of B must not be higher (= must be strictly lower) than the upper boundary of A (B entirely outside of A to the right)
        # and the upper boundary of B must not be lower (= must be strictly higher) than the lower boundary of A (B  entirely outside of A to the left)
        plinds = np.logical_and(focals[:,0] < ubin, 
                                focals[:,1] > lbin, )
        pl[i] = np.sum(masses[plinds])

        # get all sets B that are a superset of A   (=A is a subset of B)
        # that means the lower boundary of A must be greater or equal to the lower boundary of B
        # and similarly for the upper boundary
        qinds = np.logical_and(focals[:,0] <= lbin,
                               focals[:,1] >= ubin)   
        q[i] = np.sum(masses[qinds])

    bel[-1]=bel[-2]
    pl[-1]=pl[-2]
    q[-1]=q[-2]
    
    return bins, bel, pl, q

def aggregate_mass(focals_stats, hyc_mass, nbin_fact=1, cum_mass=False):
     
    n_stat, n_hyc, _ = focals_stats.shape 
    n_bins_bel = np.ceil(np.sqrt(n_hyc) * nbin_fact).astype(int)
    bins_bel = np.linspace(0, np.nanmax(focals_stats), n_bins_bel)
    bel_stats = np.empty((n_stat, n_bins_bel))
    pl_stats = np.empty((n_stat, n_bins_bel))
    q_stats = np.empty((n_stat, n_bins_bel))    
    for i_bin in range(n_stat):
    
        # compute belief, plausibility & commonality -belief functions for each aleatory bin
        _, bel,pl,q = compute_belief(focals_stats[i_bin, :, :], hyc_mass, cumulative=cum_mass, bins=bins_bel)
        bel_stats[i_bin, :] = bel
        pl_stats[i_bin, :] = pl
        q_stats[i_bin, :] = q

    return bel_stats, pl_stats, q_stats


 
class PolyUQ(object):
    
    def __init__(self, vars_ale, vars_epi, dim_ex='cartesian'):
        '''
        primary / secondary (hyper-) variables
        
        to take into account variables that share samples they can be named equally
        e.g. to create a a precise variable in an imprecision context
        
        Parameters:
        -----------
            vars_epi: list of epistemic variables (objects of UncertainVariable class) 
                   (uncertainty quantification by interval optimization [approximated by QMCS])
            vars_ale: list of aleatory variables (objects of UncertainVariable class) 
                   (uncertainty quantification by quasi Monte-Carlo methods)
        '''
        for var in vars_ale:
            assert isinstance(var, UncertainVariable)
        for var in vars_epi:
            assert isinstance(var, UncertainVariable)
        
        assert dim_ex in ['cartesian', 'hadamard', 'vacuous']
        
        # freeze the variable sets in tuples, to avoid later alterations which might break all kinds of things
        self.vars_ale = tuple(vars_ale)
        self.vars_epi = tuple(vars_epi)
        self.vars_imp = tuple([var for var in vars_epi if var.primary])
        self.vars_inc = tuple([var for var in vars_epi if not var.primary])
        
        self.dim_ex = dim_ex
        
    def sample_qmc(self, N_mcs_ale=1000000, N_mcs_epi=100, percentiles=(0.0001, 0.9999), seed=None, check_sample_sizes=True):
        '''
        A function to generate quasi Monte Carlo samples for mixed aleatory-epistemic uncertainty quantification
        
        A Halton sequence of all variables is generated over the support within 
        user-defined percentiles of each variable. The computation of support 
        takes into account polymorphic uncertainties. For details see doc for 
        UncertainVariable and their subclasses. 
        
        Input sequences are stored separately as class variables according to the 
        type of input variables:
            * primary, both aleatory and epistemic, pd.DataFrame (N_vars_prim, max(N_mcs_*))
            * secondary epistemic (hypervariables), pd.DataFrame (N_vars_epi_noprim, max(N_mcs_*))
            * secondary aleatory (hypervariable), pd.DataFrame (N_vars_ale_noprim, max(N_mcs_*))
        Only primary input samples are needed for uncertainty propagation. 
        Secondary / hyper-variables can be treated in post-processing as the 
        mapping function does not depend directly on them. The respective input
        sequences may be used in post-processing where sampling methods are 
        employed to ensure consistent and "independent" low-discrepancy sequences.
        
        Equally named variables may share their samples, e.g. for allowing precise
        inputs in an imprecision context. However, that may lead to unexpected 
        results, when sampling. The support of an imprecise variable is the 
        support of the incremental base variable +- the support of the incremental
        delta variables. Sampling the imprecise variable over this support and 
        then using these samples later to freeze the incremental base variable 
        with the same name effectively extents the support beyond the  specified 
        percentiles. However, sampling over the support of the base variable (child_var) 
        effectively reduces the support of the imprecise variable. Another alternative
        would be to take the above into account when computing eistemic hypercube
        extents, which would add a lot of error-prone logic. -> Just live with
        this minor imperfection.
        
        Probabilities of stochastic samples may be integrated numerically
        from the RandomVariables' probability densities using the given input samples.
        
        UNTESTED:
        To enrich a sample set this function can simply be called again, with identical inputs. Ensure 
        to provide the same seed, as stored as class variable from the previous function call.

        Parameters:
        -----------
            N_mcs_* : int
                number of quasi Monte Carlo Samples to use
            percentiles: 2-tuple
                the percentiles to use for computing support boundaries of input variables
            seed: int
                a random seed to initialize the random number generator to enable
                repeatable sequence generation and / or enrichment of input samples
            check_sample_size: bool
                Whether to check the sample sizes for approximation of each epistemic
                hypercube (might be time consuming for a high number of stochastic samples)
        '''
        vars_epi = self.vars_epi
        vars_ale = self.vars_ale 
        
        vars_ale_prim = [var for var in vars_ale if var.primary]
        vars_epi_prim = [var for var in vars_epi if var.primary]
        
        all_vars_prim = vars_ale_prim + vars_epi_prim # needed to fix positions of variables for indexing/assigning/accessing corresponding sequences
        
        def traverse_children(var):
            # recursively add all child variables to the all_vars dictionary
            for child_var  in var._children:
                if not isinstance(child_var, UncertainVariable): continue
                if not child_var.name in all_vars:
                    all_vars[child_var.name] = child_var
                traverse_children(child_var)
            return
        
        all_vars = {}
        # taking into account variables that share samples / are named equally
        for var in all_vars_prim:
            all_vars[var.name] = var
            traverse_children(var)
        
        all_vars = list(all_vars.values()) 
             
        # get (truncated) support (define upper and lower bounds, e.g. 99.99 % quantiles, 0.01% quantiles) 
        supports = [var.support(percentiles) for var in all_vars]
        # print(supports, all_vars)
    
        # define equivalent uniform distributions
        vars_unif =[scipy.stats.uniform(supp[0], supp[1] - supp[0]) for supp in supports] 
    
        # sampling parameters
        N_vars = len(all_vars)
    
        N_mcs = max(N_mcs_ale,N_mcs_epi)
        
        print("Generating low-discrepancy sequences for all variables... ")
        # sample N_mcs samples from a Halton Sequence and transform to uniform bounds
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        else:
            print("Warning: The usage of seeds is currently untested.")
            
        seed_seq = np.random.SeedSequence(seed).spawn(1)[0]
        seed_seq = np.random.default_rng(seed_seq)
        engine = scipy.stats.qmc.Halton(N_vars, seed=seed_seq)
        samples = engine.random(N_mcs)
        samples = pd.DataFrame(samples, columns=[var.name for var in all_vars])
        
        for i,var in enumerate(all_vars):
            samples[var.name] = vars_unif[i].ppf(samples[var.name])
        
        inp_samp_prim = samples[[var.name for var in all_vars_prim]]
        inp_suppl_ale = samples.iloc[:N_mcs_ale][[var.name for var in vars_ale if not var.primary]]
        inp_suppl_epi = samples.iloc[:N_mcs_epi][[var.name for var in vars_epi if not var.primary]]
        
        # check the number of samples per focal set
        if check_sample_sizes:
            self.check_sample_sizes(vars_epi_prim, samples, N_mcs_ale, N_mcs_epi)
        
        self.N_mcs_ale = N_mcs_ale
        self.N_mcs_epi = N_mcs_epi
        
        self.percentiles = percentiles
        self.seed = seed
        
        self.inp_samp_prim = inp_samp_prim
        self.inp_suppl_ale = inp_suppl_ale
        self.inp_suppl_epi = inp_suppl_epi
        
        return
    
    
    def propagate(self, mapping, arg_vars):
        '''
        An input sample lattice using aleatory input samples in rows and 
        epistemic input samples in columns is used for uncertainty propagation.
        The output sample lattice can be used for stochastic/statistical analysis, 
        approximate interval optimization, sensitivitz analyses or surrogate modeling. 
        
        Parameters:
        -----------
            mapping: callable
                that takes keyword arguments as specified in arg_vars
            arg_vars: dictionary 
                mapping the function arguments (keys) to the variable names (values)
         
        Returns:
        --------
            out: np.ndarray (N_mcs_ale, N_mcs_epi)
                sequence of output samples
        '''
        # hypervariables (polymorphic outer variables) are only relevant in post-processing
        
        vars_epi = self.vars_epi
        vars_ale = self.vars_ale 
        N_mcs_ale = self.N_mcs_ale
        N_mcs_epi = self.N_mcs_epi
        
        inp_samp_prim = self.inp_samp_prim
        
        # determine, if primary variables are all the same type, or if mixed type inputs are present
        loop_ale = np.any([var.primary for var in vars_ale])
        loop_epi = np.any([var.primary for var in vars_epi])
        
        # generate grid indices
        print("Generating grid...")
        if loop_ale and loop_epi:
            inds_ale, inds_epi = np.mgrid[0:N_mcs_ale:1, 0:N_mcs_epi:1]
        elif loop_epi:
            inds_ale, inds_epi = np.mgrid[0:1:1, 0:N_mcs_epi:1]
        elif loop_ale:
            inds_ale, inds_epi = np.mgrid[0:N_mcs_ale:1, 0:1:1]
        else: 
            raise RuntimeError("No primary variables defined.")
        inds_alef = inds_ale.flatten()
        inds_epif = inds_epi.flatten()
    
        # propagation
        print("Propagating mapping function...")
        all_vars = vars_ale + vars_epi
        for var_name in arg_vars.values():
            for var in all_vars:
                if var.primary and var.name == var_name: break
            else: 
                raise RuntimeError(f'Variable {var} is not marked as primary but used as a function argument')
        for var in all_vars:
            if not var.primary: continue
            for var_name in arg_vars.values():
                if var.name == var_name: break
            else: 
                raise RuntimeError(f'Variable {var} is marked as primary but not used as a function argument')
        
        names_ale = [var.name for var in vars_ale if var.primary]
        names_epi = [var.name for var in vars_epi if var.primary]
        
        if loop_ale and loop_epi:
            out_samp = np.zeros((N_mcs_ale, N_mcs_epi))
        elif loop_ale:
            out_samp = np.zeros((N_mcs_ale, 1))
        elif loop_epi:
            out_samp = np.zeros((1, N_mcs_epi))
        
        arg_vars_ale = {arg:var for arg,var in arg_vars.items() if var in names_ale}
        arg_vars_epi = {arg:var for arg,var in arg_vars.items() if var in names_epi}
        
        inp_samp_ale = inp_samp_prim[list(arg_vars_ale.values())].values
        inp_samp_epi = inp_samp_prim[list(arg_vars_epi.values())].values
        
        fcount=0
        for ind_ale, ind_epi in np.nditer([inds_alef, inds_epif]):
            # this_out = mapping(**{arg:inp_samp_prim.iloc[ind_ale][var] for arg, var in arg_vars.items() if var in names_ale}, 
            #                    **{arg:inp_samp_prim.iloc[ind_epi][var] for arg, var in arg_vars.items() if var in names_epi},)
            
            this_out = mapping(**{arg:inp_samp_ale[ind_ale, ind_var] for ind_var, arg in enumerate(arg_vars_ale.keys())}, 
                               **{arg:inp_samp_epi[ind_epi, ind_var] for ind_var, arg in enumerate(arg_vars_epi.keys())},)
            fcount += 1
            if loop_ale and loop_epi:
                out_samp[ind_ale, ind_epi] = this_out
            elif loop_epi:
                out_samp[0, ind_epi] = this_out
            elif loop_ale:
                out_samp[ind_ale, 0] = this_out
                
        if loop_ale and loop_epi:
            print(f'Mapping function was called {fcount} times in a mixed aleatory epistemic loop.')
        elif loop_ale:
            print(f'Mapping function was called {fcount} times in a pure aleatory loop.')
        elif loop_epi:
            print(f'Mapping function was called {fcount} times in a pure epistemic loop.')
            
        print("Done... ")
        self.loop_ale = loop_ale
        self.loop_epi = loop_epi
        self.out_samp = out_samp
        self.fcount = fcount
        return out_samp#, p_weights
    
    def probabilities_imp(self, i_imp=None):
        
        # many secondary aleatory variables may be provided and sampled
        # however they may not necessarily be used 
        #     e.g. two different focal sets to account for imprecision may be provided for an input variable
        #     both focal sets are bound by a different pair of aleatory variables
        #     computing the joint pdf of all aleatory variables is wrong since for each focal set only a pair of two is used
        # probability weights of primary aleatory variables
        #     if non-polymorphic -> can be readily computed
        #     if polymorphic -> depend on epistemic samples
        # for imprecise variables, probabilities are not changing, with epistemic values 
        # -> p_weights are independent, not changing within epistemic loop
        
        # pass i_imp to compute weights only for imprecise hypercube number i_imp
        
        print("Computing aleatory probability weights...")
        # compute probabilities for approximately equally spaced (due to low-discrepancy sampling) bins
        # TODO: theoretically integration has to be performed
        
        # Variables
        vars_ale = self.vars_ale
        hyc_hyp_vars = self.hyc_hyp_vars # no-imp: probably empty
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        
        if i_imp is None:
            n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
            it_imp = range(n_imp_hyc)
        else:
            n_imp_hyc = 1
            it_imp = [i_imp]
        
        # Samples
        N_mcs_ale = self.N_mcs_ale
        
        inp_samp_prim = self.inp_samp_prim.iloc[:N_mcs_ale]
        inp_suppl_ale = self.inp_suppl_ale
        
        # compute probabilities for imprecise samples
        p_weights = np.ones((N_mcs_ale, n_imp_hyc))
            
        # assemble probability weights from primary aleatory variables 
        for var in vars_ale:
            if var.primary:
                # assign weight to to all hypercubes
                p_weights *= np.repeat(var.prob_dens(inp_samp_prim[var.name])[:,np.newaxis], n_imp_hyc, axis=1)
            
        # probabilities are computed for pre-computed stochastic samples as the product of PDFs of the underlying RVs
        # each hypercube may be constructed from different hypervariables (RVs) -> has a different product probability
        # assemble probability weights from hypervariables (secondary)       
               
        hyp_dens = {}    
        for i_weight, i_imp_hyc in enumerate(it_imp):             
            hyp_vars = hyc_hyp_vars[i_imp_hyc]
            for hyp_var in hyp_vars:
                if not isinstance(hyp_var, UncertainVariable): continue
                # p_weights[:, i_imp_hyc] *= hyp_var.prob_dens(inp_suppl_ale[hyp_var.name])
                if not hyp_var.name in hyp_dens: 
                    # caching probability densities if a variable is hypervariable of multiple variables
                    hyp_dens[hyp_var.name] = hyp_var.prob_dens(inp_suppl_ale[hyp_var.name])
                p_weights[:, i_weight] *= hyp_dens[hyp_var.name]
            # normalize
            p_weights[:, i_weight] /= np.sum(p_weights[:, i_weight])
        
        return p_weights
    
    def estimate_imp(self, check_sample_sizes=True, fig=None, xvar='', yvar=''):
        
        '''    
        Estimate imprecise (quasi) Monte Carlo Samples from pre-computed mapping
        outputs
        
        
        OLD OLD OLD
        Algorithm:  
        Input: epistemic and aleatory variables
               pre-computed output sample lattice
               
        Output:
            (discrete) belief functions for each statistic
            
        
        1.) pre-assemble samples for incomplete interval optimization
        for each incompleteness sample
            freeze incomplete variables
            assemble primary p_weights
            for each aleatory sample:
                freeze aleatory hypervariables (secondary)
                assemble imprecise hypercube_sample_indices
                approximate out intervals for each imprecise hypercube
                assemble p_weights for each hypercube from hypervariables
                
            compute the statistic for each imprecise hypercube (if no imprecision there is only a single hypercube with equal upper and lower boundaries)
        
        2.) approximate incomplete interval optimization    
        for each incompleteness hypercube and each statistic and each imprecision hypercube
            approximate intervals
        
        3.) consolidate hypercubes
        for each statistic
            compute bel, pl and q
        '''
        # Variables
        vars_ale = self.vars_ale
        vars_imp = self.vars_imp
        loop_ale = self.loop_ale
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
        
        # Samples
        N_mcs_ale = self.N_mcs_ale
        N_mcs_epi = self.N_mcs_epi
        if vars_imp:
            N_mcs_imp = N_mcs_epi
        else:
            N_mcs_imp = 1
            # should be zero, but that would make hypercube_sample_indices empty
            # and subsequently break approximate_out_intervals
        
        inp_samp_prim = self.inp_samp_prim
        inp_suppl_ale = self.inp_suppl_ale
        
        out_samp = self.out_samp # Here we may choose from multiple output quantities, cluster output, etc.
        
        if fig is not None:
            from matplotlib.patches import Rectangle
            nrows = np.ceil(np.sqrt(n_imp_hyc)).astype(int)
            ncols = np.ceil(n_imp_hyc/nrows).astype(int)
            axes = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False).ravel()
            
            for var in vars_ale+vars_imp:
                if var.name == xvar:
                    xl, xu = var.support(self.percentiles)
                elif var.name == yvar:
                    yl, yu = var.support(self.percentiles)
                    
            for ax in axes:
                ax.add_patch(Rectangle((xl, yl), xu - xl, yu - yl, color='red', alpha=0.25))
                ax.plot(inp_samp_prim[xvar], inp_samp_prim[yvar],ls='none',marker=',')
                if ax.is_last_row(): ax.set_xlabel(xvar)
                if ax.is_first_col(): ax.set_ylabel(yvar)

        '1. Quantify Imprecision for each aleatory sample'
        # quantifiying imprecise QMC samples for all stochastic samples             
        # allocate arrays for interval optimization and subsequent statistical analyses
        imp_foc = np.full((N_mcs_ale, n_imp_hyc, 2), np.nan)
        if check_sample_sizes:
            sample_sizes = np.empty((N_mcs_ale, n_imp_hyc))
        pbar = simplePbar(N_mcs_ale)
        for n_ale in range(N_mcs_ale):
            # each supplementary aleatory sample defines boundaries on imprecise variables
            #    do interval optimization using the pre-computed samples within these boundaries
            #    (pre-computed epistemic samples may be the same for each aleatory sample while only imprecise input boundaries differ)
            # print(n_ale)
            this_inp_suppl = inp_suppl_ale.iloc[n_ale]
            
            if loop_ale:
                this_out = out_samp[n_ale, :]
            else:
                this_out = out_samp[0, :]
            
            # freeze the aleatory variables
            for var in vars_ale:
                if not var.primary:
                    var.freeze(this_inp_suppl[var.name])
                    # if n_ale==5: print(var, var.value)
            
            # find the indices of all epistemic samples that are within the boundaries defined by the stochastic sample
            hyc_dat_inds = self.hypercube_sample_indices(inp_samp_prim, vars_imp, imp_hyc_foc_inds, N_mcs_imp) # no-imp: np.ones(1, N_mcs_imp)
            
            if check_sample_sizes:
                hyc_num_elems = np.sum(hyc_dat_inds, axis=1)
                sample_sizes[n_ale,:] = hyc_num_elems   
            
            # compute output intervals / focal sets for each imprecise hypercube
            imp_foc[n_ale, :, :] = approximate_out_intervals(this_out, hyc_dat_inds) # no-imp: [np.min(output), np.max(output)] where min==max
            
            if fig is not None:
                # plot the hypercube part of xvar,yvar as a rectangle over the sampled output points
                numeric_focals = [var.numeric_focals for var in vars_imp]
                names_imp = [var.name for var in vars_imp]
                if names_imp:
                    xind = names_imp.index(xvar)
                    yind = names_imp.index(yvar)
                    for ax, foc_inds in zip(list(axes), imp_hyc_foc_inds):
                        xl, xu = numeric_focals[xind][foc_inds[xind]]
                        yl, yu = numeric_focals[yind][foc_inds[yind]]
                        ax.add_patch(Rectangle((xl, yl), xu - xl, yu - yl, color='grey', alpha=0.5))
                        # ax.annotate(f'{n_ale}',((xl+xu)/2,(yl+yu)/2))
            next(pbar)
            
        if check_sample_sizes:
            for i_imp_hyc in range(n_imp_hyc):
                
                max_samples = np.max(sample_sizes[:,i_imp_hyc])
                if max_samples:
                    bins = [10 ** i for i in range(-1, int(np.ceil(np.log10(max_samples))) + 1)]
                    bins[0]=0
                    bins[-1] = max_samples
                    hist, bins = np.histogram(sample_sizes[:,i_imp_hyc],bins=bins)
                else:
                    hist, bins = [1,], [0, sample_sizes[0, i_imp_hyc]]
            
                print(f'Imprecise hypercube {i_imp_hyc} epistemic sample size distribution:')
                for lbin, ubin, count in zip(bins[:-1], bins[1:], hist):
                    if count == 0: continue
                    print(f'{lbin} < n < {ubin}: {count} aleatory samples')
        
        # for no-imp: imp_foc[n_ale, 0, :] = [ np.min(output[n_ale]), np.max(output[n_ale])] where min==max
        
        self.imp_foc = imp_foc
        return imp_foc
    
    def estimate_inc(self, stat_fun, n_stat, stat_fun_kwargs={}, check_sample_sizes=True):
        '''
        Quantify incompleteness and reduce variability by applying a statistic.
        
        
        Parameters:
        ----------
            stat_fun: function
                A callable that takes a data of shape (N_mcs_ale,) and another array of weights (same shape)
                to estimate the i_stat'th of n_stat statistics (e.g. failure probability, mean and confidence bounds, maximum likelihood distribution parameters, etc.)
            n_stat: int
                the number of output arguments of stat_fun (technically it could be deduced)
            stat_fun_kwargs: dict
                any additionally arguments, needed by stat_fun
            check_sample_size: bool
                Whether to check the sample sizes for approximation of each epistemic
                hypercube (might be time consuming for a high number of stochastic samples)
            
                
        Returns:
        --------
            focals_stats: ndarray (n_hyc, n_stat)
                an array holding the focal sets / intervals for each statistic and combined epistemic hypercube
        
        '''
        # Samples
        imp_foc = self.imp_foc
        
        # Variables
        vars_inc = self.vars_inc
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
        imp_hyc_mass = self.imp_hyc_mass # no-imp: is a list containing a single 1.0
        
        inc_hyc_foc_inds = self.inc_hyc_foc_inds
        n_inc_hyc = len(inc_hyc_foc_inds)
        inc_hyc_mass = self.inc_hyc_mass 
        # no-inc: sizes are analogously to the no-imp case
        
        n_hyc = n_imp_hyc * n_inc_hyc
        
        # Samples
        N_mcs_ale = self.N_mcs_ale

        if vars_inc:
            N_mcs_inc = self.N_mcs_epi
        else:
            N_mcs_inc = 1
            
        inp_suppl_epi = self.inp_suppl_epi   
        # check the number of samples per focal set
        if check_sample_sizes:
            self.check_sample_sizes(vars_inc, inp_suppl_epi, N_mcs_ale, N_mcs_inc)

        '3.a) sample incompleteness while quantifying variability'
        # walk over supplementary incompleteness samples
        # alternatively the below could be done using bounded optimization over incomplete hypercubes
        # imp_inc_prob = np.empty((N_mcs_inc, N_mcs_ale, n_imp_hyc))
        imp_stat_samp =  np.empty((N_mcs_inc, n_stat, n_imp_hyc, 2))
        pbar = simplePbar(N_mcs_inc)
        for n_inc in range(N_mcs_inc):
            # incompleteness changes only the probability weights for already existing aleatory samples or aleatory imprecise intervals

            # freeze the epistemic variables describing incompleteness
            for var in vars_inc:
                var.freeze(inp_suppl_epi[var.name].iloc[n_inc])
            with HiddenPrints():
                p_weights = self.probabilities_imp() 
            # for no-imp: p_weights[n_ale, 0] = <product of pdfs> 

            # compute PDF/CDF for each interval boundary in each hypercube        
            # compute statistic(s) for each boundary and hypercube
            for i_imp_hyc in range(n_imp_hyc):
                for high_low in range(2):
                    
                    stat = stat_fun(imp_foc[:, i_imp_hyc, high_low], p_weights[:, i_imp_hyc], None, **stat_fun_kwargs)                    
                    imp_stat_samp[n_inc, :, i_imp_hyc, high_low] = stat
            next(pbar)
                    
        '''
        histogram gives sample counts per bin
        consider a single bin and computing histograms of the upper and lower boundaries of random intervals
        the upper boundaries will always have higher values than their lower counterparts
        there may be more/less lower/higher boundaries in a single bin than their counterparts
        that may also switch for non-cumulative sample counts at the point where pdfs overlap
        
        a range of sample counts per histogram bin is desired -> it's possibly safe to        
        sort the sample count intervals
        
        TODO: think about sorting other statistics
        '''
        imp_stat_samp.sort(axis=3)
            
        '3.b) quantify incompleteness '
        
        
        hyc_dat_inds = self.hypercube_sample_indices(inp_suppl_epi, vars_inc, inc_hyc_foc_inds, N_mcs_inc)
        
        hyc_num_elems = np.sum(hyc_dat_inds, axis=1)
        print(f"Incompleteness hypercube sample sizes: {hyc_num_elems}")   
        
        hyc_mass = np.empty((n_hyc,))
        
        # compute belief functions for each statistic
        focals_stats = np.empty((n_stat, n_hyc, 2))
        pbar = simplePbar(n_imp_hyc * n_stat)
        for i_imp_hyc in range(n_imp_hyc):
            for i_bin in range(n_stat):
                
                # compute mass-belief functions over sample counts / probability densities
                # print(i_imp_hyc, imp_stat_samp[:, i_bin, i_imp_hyc, :])
                hyc_out_focals_l = approximate_out_intervals(imp_stat_samp[:, i_bin, i_imp_hyc, 0], hyc_dat_inds) # shape (n_inc_hyc, 2)
                hyc_out_focals_u = approximate_out_intervals(imp_stat_samp[:, i_bin, i_imp_hyc, 1], hyc_dat_inds)
                
                hyc_out_focals = np.hstack((hyc_out_focals_l, hyc_out_focals_u))
                # print(hyc_out_focals.shape, hyc_out_focals, )
                
                focals_stats[i_bin, i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc , 0] = np.min(hyc_out_focals, axis=1)
                focals_stats[i_bin, i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc , 1] = np.max(hyc_out_focals, axis=1)
                # print(focals_stats[i_bin, i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc , :])
                next(pbar)
                
            hyc_mass[i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc ] = imp_hyc_mass[i_imp_hyc] * inc_hyc_mass
            # print(hyc_mass[i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc ])
            
        return focals_stats, hyc_mass
    
    def optimize_inc(self, stat_fun, n_stat, stat_fun_kwargs={}):
        
        '''

        
        '''
        
        def wrapper(x, min_max, stat_fun, samples, i_imp_hyc, i_stat, stat_fun_kwargs):
            '''
            optimization has to be done for each statistic, boundary, imprecise and incomplete hypercube separately
            i.e. a scalar value must be returned from wrapper
            
            this creates a huge overhead: 
                p_weights is computed for all imp_hyc at once -> resolved
                stat_fun returns all statistics at once
                all but one entry of these are discarded -> resolved
            
            nevertheless it might be needed for verification purposes
            
            '''
            vars_inc = self.vars_inc
            for i, var in enumerate(vars_inc):
                var.freeze(x[i])
            with HiddenPrints():
                p_weights = self.probabilities_imp(i_imp_hyc)
            stat = stat_fun(samples, p_weights, i_stat, **stat_fun_kwargs)
            return min_max * stat 
        
        
        # Samples
        imp_foc = self.imp_foc
        
        # Variables
        vars_inc = self.vars_inc
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
        imp_hyc_mass = self.imp_hyc_mass # no-imp: is a list containing a single 1.0
        
        inc_hyc_foc_inds = self.inc_hyc_foc_inds
        n_inc_hyc = len(inc_hyc_foc_inds)
        inc_hyc_mass = self.inc_hyc_mass 
        # no-inc: sizes are analogous to the no-imp case
        
        n_hyc = n_imp_hyc * n_inc_hyc
             
        numeric_focals = [var.numeric_focals for var in vars_inc]
        
        # compute belief functions for each statistic
        focals_stats = np.empty((n_stat, n_hyc, 2))
        hyc_mass = np.empty((n_hyc,))
        
        if vars_inc:
            pbar = simplePbar(n_imp_hyc*n_inc_hyc*n_stat)
            for i_imp_hyc in range(n_imp_hyc):
                for i_inc_hyc in range(n_inc_hyc):
                    i_hyc = i_imp_hyc * n_inc_hyc + i_inc_hyc
                    
                    bounds = [focs[ind] for focs, ind in zip(numeric_focals, inc_hyc_foc_inds[i_inc_hyc])]
                    init = [np.mean(bound) for bound in bounds]
                    for i_stat in range(n_stat):
                        # lower boundary
                        resll = scipy.optimize.minimize(wrapper, init, ( 1, stat_fun, imp_foc[:,i_imp_hyc, 0], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        resul = scipy.optimize.minimize(wrapper, init, (-1, stat_fun, imp_foc[:,i_imp_hyc, 0], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        # high boundary
                        reslu = scipy.optimize.minimize(wrapper, init, ( 1, stat_fun, imp_foc[:,i_imp_hyc, 1], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        resuu = scipy.optimize.minimize(wrapper, init, (-1, stat_fun, imp_foc[:,i_imp_hyc, 1], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        
                        focals_stats[i_stat, i_hyc, 0] = min( resll.fun,  reslu.fun)
                        focals_stats[i_stat, i_hyc, 1] = max(-resul.fun, -resuu.fun)
                
                        next(pbar)
                        
                hyc_mass[i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc ] = inc_hyc_mass * imp_hyc_mass[i_imp_hyc] 
        else: #no incompleteness
            with HiddenPrints():
                p_weights = self.probabilities_imp() 
            for i_imp_hyc in range(n_imp_hyc):
                for high_low in range(2):
                    stat = stat_fun(imp_foc[:, i_imp_hyc, high_low], p_weights[:, i_imp_hyc], None, **stat_fun_kwargs)                    
                    focals_stats[:, i_imp_hyc, high_low] = stat
            hyc_mass = imp_hyc_mass
            
        return focals_stats, hyc_mass
    
    def check_sample_sizes(self, vars_epi, samples, N_mcs_ale=None, N_mcs_epi=None):
        '''
        A function to check sample sizes for approximation of epistemic hypercubes
        '''
        vars_ale = self.vars_ale
        if N_mcs_ale is None:
            N_mcs_ale = samples.shape[0]
        if N_mcs_epi is None:
            N_mcs_epi = samples.shape[0]
        
        
        print("Checking sample sizes for approximation of epistemic focal sets...")
        for var in vars_epi:
            if var.is_poly:
                sample_sizes=np.empty((N_mcs_ale, var.n_focals), dtype=int)
                for n_ale in range(N_mcs_ale):
                    for var_ale in vars_ale:
                        var_ale.freeze(samples[var_ale.name].iloc[n_ale])
                    for i_foc, (boundl, boundr) in enumerate(var.numeric_focals):
                        this_selector = np.logical_and(samples[var.name].iloc[:N_mcs_epi]>=boundl,
                                                       samples[var.name].iloc[:N_mcs_epi]<=boundr)
                        sample_sizes[n_ale, i_foc] = np.sum(this_selector)
            else:
                sample_sizes=np.empty((1, var.n_focals), dtype=int)
                for i_foc, (boundl, boundr) in enumerate(var.numeric_focals):
                    this_selector = np.logical_and(samples[var.name].iloc[:N_mcs_epi]>=boundl,
                                                   samples[var.name].iloc[:N_mcs_epi]<=boundr)
                    sample_sizes[0, i_foc] = np.sum(this_selector)
                    # print(f'Focal set {i_foc} of variable {var.name} contains {sample_sizes[0, i_foc]} epistemic samples.')
            for i_foc in range(var.n_focals):
                max_samples = np.max(sample_sizes[:,i_foc])
                if var.is_poly and max_samples:
                    bins = [10 ** i for i in range(-1, int(np.ceil(np.log10(max_samples))) + 1)]
                    bins[0]=0
                    #bins[1]=1
                    bins[-1] = max_samples
                    hist, bins = np.histogram(sample_sizes[:,i_foc],bins=bins)
                else:
                    hist, bins = [1,], [sample_sizes[0, i_foc], sample_sizes[0, i_foc]]
                    
                print(f'Focal set {i_foc} of variable {var} epistemic sample size distribution:')
                for lbin, ubin, count in zip(bins[:-1], bins[1:], hist):
                    if count == 0: continue
                    print(f'{lbin} < n < {ubin}: {count} aleatory samples')
                
    @property
    def hyc_hyp_vars(self, ):
        # each imprecise hypercube may be constructed from different 
        # aleatory hypervariables (RVs) -> has a different product probability
        
        # hyc_hyp_var indices refer to imprecise hypercubes
        
        vars_imp = self.vars_imp
        hyc_foc_inds = self.imp_hyc_foc_inds
        n_hyc = len(hyc_foc_inds)
        
        #assemble hypervariables per hypercube (secondary)
        hyc_hyp_vars = [set() for i_hyc in range(n_hyc)] # use sets to avoid double counting of variables that may be hypervariables of more than one focal set
        for i_hyc in range(n_hyc):
            # walk over primary epistemic variables and their hypervariables  
            for i_epi, var_epi in enumerate(vars_imp):
                # skip if hypercube is defined by fixed values -> stochastic probabilities are not changing
                if not var_epi.is_poly: continue
                
                hyp_vars = var_epi._focals[hyc_foc_inds[i_hyc][i_epi]]
                hyc_hyp_vars[i_hyc].update(hyp_vars)      
        
        return hyc_hyp_vars

    def hyc_foc_inds(self, vars_):
        
        if self.dim_ex=='cartesian':
            # cartesian product of input intervals / focal sets
            # for each hypercube: indices of focal sets of each variable
            return list(product(*[range(var.n_focals) for var in vars_]))
        elif self.dim_ex=='hadamard':
            # for fuzzy sets, an alpha level would be chosen, which results in a single focal set per variable and thus a single hypercube
            # each variable must have the same number of focal sets
            n_focals = vars_[0].n_focals
            n_vars = len(vars_)
            return [[i_foc, ] * n_vars for i_foc in range(n_focals)]
        elif self.dim_ex=='vacuous':
            raise NotImplementedError('Vacuous extension currently not implemented')
        else:
            raise ValueError(f'Dimension extension {self.dim_ex} not understood.')
        
    @property
    def imp_hyc_foc_inds(self,):
        vars_imp = self.vars_imp
        return self.hyc_foc_inds(vars_imp)
    
    @property
    def inc_hyc_foc_inds(self):
        vars_inc = self.vars_inc
        return self.hyc_foc_inds(vars_inc)
        
    
    def hyc_mass(self, vars_):
        hyc_foc_inds = self.hyc_foc_inds(vars_)
        n_hyc = len(hyc_foc_inds)
        
        hyc_mass = np.empty((n_hyc,))
        for i_hyc, foc_inds in enumerate(hyc_foc_inds):
            # compute output prob_dens for this hypercube
            if self.dim_ex=='cartesian':
                #returns 1.0 for empty hypercube
                hyc_mass[i_hyc] = np.product([var.masses[ind] for var, ind in zip(vars_, foc_inds)])
            elif self.dim_ex=='hadamard':
                # returns nan for empty hypercube
                hyc_mass[i_hyc] = np.average([var.masses[ind] for var, ind in zip(vars_, foc_inds)])
            else:
                raise NotImplementedError(f'{self.dim_ex} dimension extension currently not implemented')
            
        return hyc_mass
    
    @property
    def imp_hyc_mass(self):
        vars_imp = self.vars_imp
        return self.hyc_mass(vars_imp)
    
    @property
    def inc_hyc_mass(self):
        vars_inc = self.vars_inc
        return self.hyc_mass(vars_inc)
    
    def hypercube_sample_indices(self, inp_samples, vars_epi, hyc_foc_inds=None, N_mcs_epi=None):
        '''
        generate the input sample indices for all hypercubes defined by
        the focal sets of the provided epistemic variables
        
        Parameters:
        vars_epi: epistemic variables to consider for the hypercubes
        inp_samples: pd.DataFrame n_vars x N_mcs_*
        hyc_foc_inds: list-of-list-of-indices for each hypercube the indices of the corresponding focal sets of each epistemic variable
        N_mcs_epi: number of epistemic input samples, inferred from inp_samples if not given 
        '''
        
        if hyc_foc_inds is None:
            hyc_foc_inds = self.hyc_foc_inds(vars_epi) # no-imp -> [()]
            
        n_hyc = len(hyc_foc_inds) # no-imp: 1
        
        if N_mcs_epi is None:
            N_mcs_epi = inp_samples.shape[0]
            
        # extract underlying numpy array in order of vars_epi for faster indexing 
        inp_samples = inp_samples[[var.name for var in vars_epi]].values[:N_mcs_epi, :]
        
        hyc_dat_inds = np.ones((n_hyc, N_mcs_epi), dtype=bool)
        numeric_focals = [var.numeric_focals for var in vars_epi] # no-imp []
        for i_hyc, hypercube in enumerate(hyc_foc_inds):
            # get focal sets / intervals
            focals = [focals[ind] for focals,ind in zip(numeric_focals, hypercube)] # no-imp []
            # select matching epistemic inp_samples
            selector = hyc_dat_inds[i_hyc,:]
            # for (boundl,boundr), var in zip(focals,vars_epi):
            #     this_selector = np.logical_and(inp_samples[var.name]>=boundl,
            #                                    inp_samples[var.name]<=boundr)
            for i_var, (boundl, boundr) in enumerate(focals):
                selector &= inp_samples[:, i_var]>=boundl
                selector &= inp_samples[:, i_var]<=boundr 
                # no-imp: [1,1,...,1,1] select all samples
                
        return hyc_dat_inds    

        
def generate_histogram_bins(data, axis=1, nbin_fact=1):
    # generate the bins
    # modified Freedman-Diaconis rule 
    # for each epistemic sample or hypercube/focal set boundary, compute the 25% and 75% quantiles
    # the axis parameter is not tested
    
    N_mcs = data.shape[int(not axis)]
    quantiles = np.apply_along_axis(np.nanquantile, axis, data, [0.75,0.25])
    # compute the maximum interquartile range over all epistemic samples
    iqr = np.max(quantiles[:,0]) - np.min(quantiles[:,1])
    # compute the bin width based on Freedman-Diaconis equation
    bin_width_dens = 2 * iqr * np.power(N_mcs, -1/3)
    # compute the bins
    start, stop = np.nanmin(data), np.nanmax(data)
    out_range = stop - start
    n_bins_dens = np.ceil(out_range / bin_width_dens * nbin_fact).astype(int)
    bins_densities = np.linspace(start, stop, n_bins_dens + 1) # that is what needs to be passed to histogram functions
    
    return bins_densities
        
            
if __name__ == '__main__':
    pass

    # mu1 = MassFunction(name='mu1', focals=[(14.78,14.80),(13.96,15.61)], masses=[0.5,0.5], primary=False)
    # sig1 = MassFunction(name='sig1', focals=[(4.16,4.18),(3.66,4.85)], masses = [0.5, 0.5], primary=False)
    # q1 = RandomVariable(name='q1', dist='norm', params=[mu1, sig1])
    # print(q1.support())
    #
    # inc_q1a1 = (RandomVariable(name='q1a1l', dist='norm', params=[15,4], primary=False),
    #           RandomVariable(name='q1a1l', dist='norm', params=[0.1,0.01], primary=False),)
    # inc_q1a0 = (RandomVariable(name='q1a1l', dist='norm', params=[15,4], primary=False),
    #             RandomVariable(name='q1a0l', dist='norm', params=[-0.4,0.1], primary=False), 
    #           RandomVariable(name='dq1a0r', dist='norm', params=[0.5,0.06], primary=False))
    # q1 = MassFunction(name='q1', focals=[inc_q1a1, inc_q1a0], masses=[0.5,0.5], primary=True, incremental=True)
    # print(q1.support())