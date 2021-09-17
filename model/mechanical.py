# from importlib import reload; from model import mechanical; mechanical.main()
# mechanical.clear_and_exit();reload(mechanical); mechanical.main()

import time
import numpy as np
import matplotlib
#matplotlib.use('qt5Agg')
import matplotlib.pyplot as plot
# plot.ioff()
import os
import sys
import glob
import shutil

import logging
# global logger
# logger = logger.getLogger('')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

import pyansys

# import scipy.stats
# import scipy.optimize
# import scipy.signal
import scipy.io
# import scipy.integrate
import uuid
import warnings
# import simpleflock

# plot.figure()
# plot.close("all")

# plot.show()
print_context_dict = {'text.usetex':True,
                     'text.latex.preamble':"\\usepackage{siunitx}\n \\usepackage{xfrac}",
                     'font.size':10,
                     'legend.fontsize':10,
                     'xtick.labelsize':10,
                     'ytick.labelsize':10,
                     'axes.labelsize':10,
                     'font.family':'serif',
                     'legend.labelspacing':0.1,
                     'axes.linewidth':0.5,
                     'xtick.major.width':0.2,
                     'ytick.major.width':0.2,
                     'xtick.major.width':0.5,
                     'ytick.major.width':0.5,
                     'figure.figsize':(5.906, 5.906 / 1.618),  # print #150 mm \columnwidth
                     # 'figure.figsize':(5.906/2,5.906/2/1.618),#print #150 mm \columnwidth
                     # 'figure.figsize':(5.53/2,2.96),#beamer
                     # 'figure.figsize':(5.53/2*2,2.96*2),#beamer
                     'figure.dpi':100}
    # figsize=(5.53,2.96)#beamer 16:9
    # figsize=(3.69,2.96)#beamer 16:9
    # plot.rc('axes.formatter',use_locale=True) #german months
# must be manually set due to some matplotlib bugs
# if print_context_dict['text.usetex']:
    # # plt.rc('text.latex',unicode=True)
    # plot.rc('text', usetex=True)
    # plot.rc('text.latex', preamble="\\usepackage{siunitx}\n \\usepackage{xfrac}")

# import math
# class LogFormatterSciNotation(matplotlib.ticker.LogFormatterMathtext):
#     """
#     Format values following scientific notation in a logarithmic axis.
#     """
#
#     def _non_decade_format(self, sign_string, base, fx, usetex):
#         'Return string for non-decade locations'
#         b = float(base)
#         exponent = math.floor(fx)
#         coeff = b ** fx / b ** exponent
#
#         coeff = round(coeff)
#         return f'$\\num{{{coeff}E{exponent}}}$'
#         return (r'$%s%g E {%d}$') % \
#                                     (sign_string, coeff, exponent)

import functools
from contextlib import contextmanager


@contextmanager
def supress_logging(mapdl):
    """Contextmanager to supress logging for a MAPDL instance"""
    prior_log_level = mapdl._log.level
    if prior_log_level != 'CRITICAL':
        mapdl._set_log_level('CRITICAL')
    try:
        yield
    finally:
        if prior_log_level != 'CRITICAL':
            mapdl._set_log_level(prior_log_level)


def session_restore(func):

    @functools.wraps(func)
    def wrapper_session_restore(*args, **kwargs):
        mech = args[0]
        ansys = mech.ansys
        routine_dict = {0.0:ansys.finish,
                        17.0:ansys.prep7,
                        21.0:ansys.slashsolu,
                        31.0:ansys.post1,
                        36.0:ansys.post26,
                        52.0:ansys.aux2 ,
                        53.0:ansys.aux3,
                        62.0:ansys.aux12,
                        65.0:ansys.aux15}
        ansys.get(par='ROUT', entity='ACTIVE', entnum='0', item1='ROUT')
        # ansys.load_parameters()
        current_rout = ansys.parameters['ROUT']
        if current_rout != 0.0:
            ansys.finish()
        value = func(*args, **kwargs)
        if current_rout != 0.0:
            ansys.finish()
            routine_dict[current_rout]()
        return value

    return wrapper_session_restore


class MechanicalDummy(object):
    
    def __init__(self, jobname):
        logger.info('Non functioning dummy of Mechanical for faster loading of previous results.')
        self.jobname = jobname
        
        #             build_mdof, free_decay, ambient, impulse_response, modal, modal_comp  , IRF matrix
        self.state = [False,      False,      False,   False,            False, False,        False]

        self.nonlinear_elements = []
        self.voigt_kelvin_elements = []
        self.coulomb_elements = []
        self.mass_elements = []
        self.beam_elements = []

        # initialize class variables
        # build_mdof
        self.nodes_coordinates = None
        self.num_nodes = None
        self.num_modes = None
        self.d_vals = None
        self.k_vals = None
        self.masses = None
        self.eps_duff_vals = None
        self.sl_force_vals = None
        self.hyst_vals = None
        self.meas_nodes = None
        self.damping = None
        
        # free_decay
        self.decay_mode = None
        self.t_vals_decay = None
        self.resp_hist_decay = None
        
        #ambient
        self.inp_hist_amb = None
        self.t_vals_amb = None
        self.resp_hist_amb = None
        
        #impulse_response
        self.inp_hist_imp = None
        self.t_vals_imp = None
        self.resp_hist_imp = None
        self.modal_imp_energies = None
        self.modal_imp_amplitudes = None
        
        # IRF matrix 
        self.t_vals_imp = None
        self.IRF_matrix = None
        self.imp_hist_imp_matrix = None
        self.modal_imp_energy_matrix = None
        self.modal_imp_amplitude_matrix = None
        
        # modal
        self.damped_frequencies = None
        self.modal_damping = None
        self.damped_mode_shapes = None
        self.frequencies = None
        self.mode_shapes = None
        self.num_modes = None
        
        #signal_parameters
        self.deltat = None
        self.timesteps = None
        
        #transient_parameters
        self.trans_params = None
        
    def build_mdof(self, nodes_coordinates=[(1, 0, 0, 0), ],
                   k_vals=[1], masses=[1], d_vals=None, damping=None,
                   sl_force_vals=None, eps_duff_vals=None, hyst_vals=None,
                   num_modes=None, meas_nodes=None, **kwargs):
        
        logger.debug('self.build_mdof')
        # if m modes are of interest number of nodes n > 10 m

        num_nodes = len(nodes_coordinates)

        # Nodes
        ntn_conns = [[] for i in range(num_nodes - 1)]
        for i, (node, x, y, z) in enumerate(nodes_coordinates):
            if i < num_nodes - 1:
                ntn_conns[i] = [node]
            if i > 0:
                ntn_conns[i - 1].append(node)

        self.nodes_coordinates = nodes_coordinates
        self.ntn_conns = ntn_conns
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        self.d_vals = d_vals
        self.k_vals = k_vals
        self.masses = masses
        self.eps_duff_vals = eps_duff_vals
        self.sl_force_vals = sl_force_vals
        self.hyst_vals = hyst_vals

        self.damping = damping
        self.meas_nodes = np.array(meas_nodes)
        
        self.state[0] = True
        for i in range(1, len(self.state)):
            self.state[i] = False

    @classmethod
    def load(cls, jobname, load_dir):
        
        assert os.path.isdir(load_dir)
        
        fname = os.path.join(load_dir, f'{jobname}_mechanical.npz')
        assert os.path.exists(fname)
        
        logger.info('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        
        assert jobname == in_dict['self.jobname'].item()
        
        mech = cls(jobname)
        
        state = list(in_dict['self.state'])
        
        if state[0]:
            logger.debug('stat[0]')
            nodes_coordinates = in_dict['self.nodes_coordinates']
            #print(nodes_coordinates, 'should be list of lists')
            #num_nodes = in_dict['self.num_nodes']
            num_modes = in_dict['self.num_modes'].item()
            d_vals = list(in_dict['self.d_vals'])
            k_vals = list(in_dict['self.k_vals'])
            masses = list(in_dict['self.masses'])
            eps_duff_vals = list(in_dict['self.eps_duff_vals'])
            sl_force_vals = list(in_dict['self.sl_force_vals'])
            hyst_vals = list(in_dict['self.hyst_vals'])
            meas_nodes = list(in_dict['self.meas_nodes'])
            #print(in_dict['self.damping'], type(in_dict['self.damping']))
            damping = in_dict['self.damping']
            if damping.size == 1:
                damping = damping.item()
            else:
                damping = list(damping)
                if len(damping) >= 2:
                    if damping[-1] == 1 or damping[-1] == 0:
                        damping[-1] = bool(damping[-1])
            
            mech.build_mdof(nodes_coordinates=nodes_coordinates,
                            k_vals=k_vals, masses=masses, d_vals=d_vals,
                            damping=damping, sl_force_vals=sl_force_vals,
                            eps_duff_vals=eps_duff_vals, hyst_vals=hyst_vals,
                            num_modes=num_modes, meas_nodes=meas_nodes)
        
        if state[1]:
            logger.debug('state[1]')
            mech.decay_mode = in_dict['self.decay_mode'].item()
            mech.t_vals_decay = in_dict['self.t_vals_decay']
            mech.resp_hist_decay = [None, None, None]
            for i, key in enumerate(['self.resp_hist_decayd',
                                     'self.resp_hist_decayv',
                                     'self.resp_hist_decaya']):
                arr = in_dict[key]
                if not arr.shape: arr = arr.item()
                mech.resp_hist_decay[i] = arr

        if state[2]:
            logger.debug('state[2]')
            mech.inp_hist_amb = in_dict['self.inp_hist_amb']
            mech.t_vals_amb = in_dict['self.t_vals_amb']
            mech.resp_hist_amb = [None, None, None]
            for i, key in enumerate(['self.resp_hist_ambd',
                                     'self.resp_hist_ambv',
                                     'self.resp_hist_amba']):
                arr = in_dict[key]
                if not arr.shape: arr = arr.item()
                mech.resp_hist_amb[i] = arr
                
        if state[3]:
            logger.debug('state[3]')
            mech.inp_hist_imp = in_dict['self.inp_hist_imp']
            mech.t_vals_imp = in_dict['self.t_vals_imp']
            mech.resp_hist_imp = [None, None, None]
            for i, key in enumerate(['self.resp_hist_impd',
                                     'self.resp_hist_impv',
                                     'self.resp_hist_impa']):
                arr = in_dict[key]
                if not arr.shape: arr = arr.item()
                mech.resp_hist_imp[i] = arr
            mech.modal_imp_energies = in_dict['self.modal_imp_energies']
            mech.modal_imp_amplitudes = in_dict['self.modal_imp_amplitudes']

        if state[4]:
            logger.debug('state[4]')
            mech.damped_frequencies = in_dict['self.damped_frequencies']
            mech.modal_damping = in_dict['self.modal_damping']
            mech.damped_mode_shapes = in_dict['self.damped_mode_shapes']
            mech.frequencies = in_dict['self.frequencies']
            mech.mode_shapes = in_dict['self.mode_shapes']
            mech.num_modes = in_dict['self.num_modes']
        
        if state[5]:
            logger.debug('state[5]')
            mech.frequencies_comp = in_dict['self.frequencies_comp']
            mech.modal_damping_comp = in_dict['self.modal_damping_comp']
            mech.mode_shapes_comp = in_dict['self.mode_shapes_comp']
            
        if state[2] or state[3] or state[4] or state[6]:
            logger.debug('state[2,3,4 or 6]')
            trans_params = in_dict['self.trans_params']
            if trans_params.size > 1:
                mech.trans_params = tuple(trans_params)
                
            mech.deltat = in_dict['self.deltat'].item()
            mech.timesteps = in_dict['self.timesteps'].item()
        
        if state[6]:
            logger.debug('state[6]')
            mech.t_vals_imp = in_dict['self.t_vals_imp']
            mech.IRF_matrix = in_dict['self.IRF_matrix']
            mech.imp_hist_imp_matrix = in_dict['self.imp_hist_imp_matrix']
            mech.modal_imp_energy_matrix = in_dict['self.modal_imp_energy_matrix']
            mech.modal_imp_amplitude_matrix = in_dict['self.modal_imp_amplitude_matrix']
        
        mech.state = state
        
        return mech


class Mechanical(object):
    #TODO: Update to newest pyANSYS release
    
    def __init__(self, ansys=None, jobname=None, wdir=None):
        
        if wdir is not None:
            if not os.path.isdir(wdir):
                os.makedirs(wdir)
        
        if ansys is None:
            ansys = self.start_ansys(wdir, jobname)
        assert isinstance(ansys, pyansys.mapdl._MapdlCore)
        
        try:
            path = os.path.join(ansys.directory, ansys.jobname)
            ansys.finish()
            ansys.clear()
            for file in glob.glob(f'{path}.*'):
                os.remove(file)
            if os.path.exists(f'{path}/'):
                for file in glob.glob(f'{path}/*'):
                    os.remove(file)
                os.rmdir(f'{path}/')
        except (pyansys.errors.MapdlExitedError, NameError) as e:
            logger.exception(e)
            ansys = self.start_ansys(wdir, jobname)
        
        self.ansys = ansys
        
        if wdir is not None:
            logger.info(f'Switching working directory:\t {wdir}')
            ansys.cwd(wdir)
        if jobname is not None:
            logger.info(f'Switching job:\t {jobname}')
            ansys.filname(fname=jobname, key=1)
            self.jobname = jobname
        else:
            logger.info(f'Current job:\t {ansys.jobname}')
            self.jobname = ansys.jobname
        ansys.finish()
        ansys.clear()
        ansys.config(lab='NOELDB', value=1)
#         ansys.config(lab='NORSTGM',value=1)
        # ansys.output(fname='null',loc='/dev/')
        #ansys.nopr()  # Suppresses the expanded interpreted input data listing. Leads to failures at least in modal
        ansys.nolist()  # Suppresses the data input listing.
        ansys.finish()
        #             build_mdof, free_decay, ambient, impulse_response, modal, modal_comp  , IRF matrix
        self.state = [False,      False,      False,   False,            False, False,        False]

        self.nonlinear_elements = []
        self.voigt_kelvin_elements = []
        self.coulomb_elements = []
        self.mass_elements = []
        self.beam_elements = []

        # initialize class variables
        # build_mdof
        self.nodes_coordinates = None
        self.num_nodes = None
        self.num_modes = None
        self.d_vals = None
        self.k_vals = None
        self.masses = None
        self.eps_duff_vals = None
        self.sl_force_vals = None
        self.hyst_vals = None
        self.meas_nodes = None
        self.damping = None
        
        # free_decay
        self.decay_mode = None
        self.t_vals_decay = None
        self.resp_hist_decay = None
        
        #ambient
        self.inp_hist_amb = None
        self.t_vals_amb = None
        self.resp_hist_amb = None
        
        #impulse_response
        self.inp_hist_imp = None
        self.t_vals_imp = None
        self.resp_hist_imp = None
        self.modal_imp_energies = None
        self.modal_imp_amplitudes = None
        
        # IRF matrix 
        self.t_vals_imp = None
        self.IRF_matrix = None
        self.imp_hist_imp_matrix = None
        self.modal_imp_energy_matrix = None
        self.modal_imp_amplitude_matrix = None
        
        # modal
        self.damped_frequencies = None
        self.modal_damping = None
        self.damped_mode_shapes = None
        self.frequencies = None
        self.mode_shapes = None
        self.num_modes = None
        
        #signal_parameters
        self.deltat = None
        self.timesteps = None
        
        #transient_parameters
        self.trans_params = None
    
    @staticmethod
    def start_ansys(working_dir=None, jid=None,):
        
        # global ansys
        # try:
            # ansys.finish()
        # except (pyansys.errors.MapdlExitedError, NameError) as e:
            # logger.warning(repr(e))
        if working_dir is None:
            working_dir = os.getcwd()
        os.chdir(working_dir)
        now = time.time()
        if jid is None:
            jid = 'file'
        ansys = pyansys.launch_mapdl(
            exec_file='/usr/app-soft/ansys/v201/ansys/bin/ansys201',
            run_location=working_dir, override=True, loglevel='ERROR',
            nproc=1, log_apdl='w',
            log_broadcast=False, jobname=jid,
            mode='console', additional_switches='-smp')
    
        logger.info(f'Took {time.time()-now} s to start up ANSYS.')
        
        ansys.clear()
        
        return ansys

        
    @session_restore
    def nonlinear(self, nl_ity=1, d_max=1.5, k_lin=1e5, k_nl=None, linear_part=False):
        '''
        generate a nonlinear spring
        with a linear part and a cubic nonlinear part
        nl_ity: the fraction of linear and nonlinear parts
            for nl_ity = (-0.5,0] it is a nonlinear softening spring,degressive, underlinear
            for nl_ity = 0 it is a linear spring
            for nl_ity = [0,0.5) it is a nonlinear hardening spring,orogressive, overlinear
        d_max: maximal displacement, up to which the force-displacement curve will be defined. ansys will interpolate further points and (probably) issue a warning.
        k_lin: linear stiffness
        k_nl: nonlinear stiffness, if not automatically determined: k_lin and k_nl should match at d_max
        '''
        if k_nl is not None:
            logger.warning('Are you sure, you want to set k_nl manually. Make sure to not overshoot d_max')
        if nl_ity < -0.5 or nl_ity > 0.5:
            raise RuntimeError('The fraction of nonlinearity should not exceed/subceed 0.5/-0.5. It currently is set to {}.'.format(nl_ity))

        ansys = self.ansys
        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        if k_nl is None:
            k_at_d_max = k_lin / d_max ** 2
            k_nl = k_at_d_max
        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='COMBIN39', kop3='3', inopr=1)
        d = np.linspace(0, d_max, 20, endpoint=True)

        if linear_part:
            F = k_lin * d * (1 - nl_ity) + k_nl * d ** 3 * (nl_ity)
        else:
            F = k_nl * d ** 3 * (nl_ity)
#         plot.plot(d,F)
#         plot.show()
        d = list(d)
        F = list(F)
        command = 'R, {}'.format(ansys.parameters['NSET'] + 1)
        i = 0
        while len(d) >= 1:
            command += ''.join(', {}, {}'.format(d.pop(0), F.pop(0)))
            i += 1
            if i == 3:
                ansys.run(command)
                # print(command)
                command = 'RMORE'
                i = 0
        else:
            ansys.run(command)
            # print(command)

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.nonlinear_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

        # ansys.r(nset=1, r1=d.pop(0), r2=F.pop(0), r3=d.pop(0), r4=F.pop(0), r5=d.pop(0), r6=F.pop(0))

    @session_restore
    def voigt_kelvin(self, k=100000, d=150, **kwargs):
        ansys = self.ansys

        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='COMBIN14', inopr=1, kop2='3')
        #              k,    cv1, cv2
        ansys.r(nset=ansys.parameters['NSET'] + 1, r1=k, r2=d)

#         Omega = 1
#         s=1
#         Wd = d*np.pi*Omega*s**2 # petersen eq. 555
#         print(f"Damping energy at unit displacement and unit circular frequency: {Wd}")

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.voigt_kelvin_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

    @session_restore
    def coulomb(self, k_1=90000, d=0, f_sl=15000, k_2=10000 , **kwargs):
        '''
        k_1 and slider are in series
        k_1 is the sticking stiffness, and defines the displacement required
        to reach the force, at which the slider breaks loose should be k_2*10


        f_sl determines the constant friction damping force that is applied each half-cycle
            a lower f_sl means less damping, longer decay
            a higher f_sl means higher damping, shorter decay
            should be d_0/k_2

        high f_sl or low k_1 have the same effect

        k_2 is the direct stiffness and in parallel with k_1+slider and
            determines the oscillation frequency if a mass is attached to the element
            apdl modal solver uses k_tot = k_1 + k_2


        '''
        ansys = self.ansys
        ansys.prep7()

        k_tot = kwargs.pop('k_tot', k_2 + k_1)
        k_2 = k_tot - k_1
        if k_tot:
            f_sl_in = f_sl * k_1 / k_tot
        else:
            f_sl_in = f_sl

#         print('Displacement amplitude above which friction force becomes active and below which no more dissipation happens {}'.format(f_sl/k_1))
#         s=1
#         Wd = 4*f_sl_in*(s-f_sl_in/k_1) # Petersen eq 637
#         print(f"Damping energy at {s} displacement: {Wd}")

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='COMBIN40', kop3='3', inopr=1)
        #               K1,     C,     M,    GAP,  FSLIDE, K2
        # print(k_1, d,f_sl_in, k_2)
        ansys.r(nset=ansys.parameters['NSET'] + 1, r1=k_1, r2=d, r3=0.0, r4=0.0, r5=f_sl_in, r6=k_2)
        #               K1,     C,     M,    GAP,  FSLIDE, K2
        # ansys.r(nset=1, r1=0, r2=1000, r3=0, r4=0, r5=0, r6=100)#
#         print(f"nset= {ansys.parameters['NSET']+1}, r1={k_1}, r2={d}, r3={0.0}, r4={0.0}, r5={f_sl_in}, r6={k_2}")
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.coulomb_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

    @session_restore
    def mass(self, m=100):
        ansys = self.ansys
        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='MASS21', inopr=1)
        ansys.r(nset=ansys.parameters['NSET'] + 1, r3=m)

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.mass_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

    def beam(self, E, PRXY, A, Iyy, Izz, Iyz,):
        ansys = self.ansys
        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='BEAM188', kop3=3, inopr=1)
#         ansys.r(nset= ansys.parameters['NSET']+1, r3=m)
        ansys.sectype(1, "BEAM", "ASEC", "tube", 0)
        '''
        Arbitrary: User-supplied integrated section
        properties instead of basic geometry data.
        Data to provide in the value fields:
        A, Iyy, Iyz, Izz, Iw, J, CGy, CGz, SHy,
        SHz, TKz, TKy
        where
        A = Area of section
        Iyy = Moment of inertia about the y axis
        Iyz = Product of inertia
        Izz = Moment of inertia about the z axis
        Iw = Warping constant
        J = Torsional constant
        CGy = y coordinate of centroid
        CGz = z coordinate of centroid
        SHy = y coordinate of shear center
        SHz = z coordinate of shear center
        TKz = Thickness along Z axis (maximum
        height)
        TKy = Thickness along Y axis (maximum
        width
        '''
        CGy = 0
        CGz = 0
        SHz = 0
        SHy = 0
        Tkz = 0
        Tky = 0
        # validate the above at some point

        IW = 0
        J = 0
        # no warping, no torsion, maybe has to be set to infty or whatever

        ansys.secdata(A, Iyy, Iyz, Izz, IW, J, CGy, CGz, SHy, SHz, Tkz, Tky)
        # SECOFFSET, CENT

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()

        self.beam_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

#     def build_conti(self, parameters, Ldiv, initial=None, meas_locs=None):
#         ansys=self.ansys
#         ansys.prep7()
#         assert Ldiv >= 3
#
#         #Nodes
#         L = parameters['L']
#         x_nodes = np.linspace(0,L,Ldiv)
#
#         x_knl = parameters['x_knl']
#         x_nodes[np.argmin(np.abs(x_nodes-x_knl))] = x_knl
#
#         if initial is not None:
#             x_d0 = initial['x_d0']
#             x_nodes[np.argmin(np.abs(x_nodes-x_d0))] = x_d0
#
#         print(x_nodes, x_knl, x_d0)
#
#         for x_node in x_nodes:
#             ansys.n(x=x_node, y=0, z=0)
#
#
#
#         #boundary conditions
#         ansys.nsel(type, item='LOC', comp='x', vmin=0, vmax=0)
#         ansys.get('bcnode','node',0,"num","min")
#         ansys.d(node='bcnode', value=0,lab='UX',lab2='UY',lab3='UZ')
#
#
#         ansys.nsel(type, item='LOC', comp='x', vmin=x_knl, vmax=x_knl)
#         ansys.get('knlnode','node',0,"num","min")
#         ansys.d(node='knlnode', value=0,lab2='UY',lab3='UZ')
#
#
#         if initial is not None:
#             x_d0 = initial['x_d0']
#             d0y = initial['d0y']
#             d0z = initial['d0z']
#
#             ansys.nsel(type, item='LOC', comp='x', vmin=x_d0, vmax=x_d0)
#             ansys.get('ininode','node',0,"num","min")
#             ansys.ic(node='ininode', lab='UY', value=d0y)
#             ansys.ic(node='ininode', lab='UZ', value=d0z)
#
#
#         if meas_locs is not None:
#             ansys.nsel(type='NONE')
#             for x_loc in meas_locs:
#                 ansys.nsel('A', item='LOC', comp='x', vmin=x_loc, vmax=x_loc)
#             ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
#         else:
#             ansys.nsel(type='ALL')
#
#         ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
#
#
#
#         ansys.finish()

    def build_mdof(self, nodes_coordinates=[(1, 0, 0, 0), ],
                   k_vals=[1], masses=[1], d_vals=None, damping=None,
                   sl_force_vals=None, eps_duff_vals=None, hyst_vals=None,
                   num_modes=None, meas_nodes=None, **kwargs):
        '''
        this function build a MDOF spring mass system

        nodes_coordinates is a list of tuples (node_number, x, y z)
            first node in nodes_coordinates is taken as a fixed support

        masses are applied to every node
        k and d are applied between first and second node, and so on

        sl_forces are the sliding forces for friction damping.
            equivalent friction depends on an assumed omega and displacement,
            therefore must be estimated outside this function

        eps_duff_vals are the factors for the cubic nonlinear springs (connection),

        meas_nodes is a list of nodes of which the transient responses should be saved

        argument:
            damping = None -> no damping
                zeta -> global rayleigh,
                (zeta, zeta) -> global rayleigh
                (alpha,beta,True/False) -> global/elementwise proportional
                (eta, True/False) -> Structural/Hysteretic (global -> only damped modal and harmonic analyses not TRANSIENT! / elementwise -> only transient (IWAN))
                (Friction -> only local, via sl_forces
                IWAN needs an equivalent damping ratio and a stress-strain relationship (G,E?)
                from which a monotonic strain-softening backbone curve \tau vs \gamma (considering shear) according to Masing's rule can be computed (Kausel Eq. 7.307),
                in turn this can be approximated/discretized by N spring-slider systems in parallel (or a nonlinear stiffness
                    TRY IN SDOF first

        TODO:
        (store constrained nodes to exclude them from indexing operations)
        (extend 3D: arguments constraints, ntn_conns)
        (extend nonlinear)
        (extend damping)

        '''
        logger.debug('self.build_mdof')
        # if m modes are of interest number of nodes n > 10 m

        num_nodes = len(nodes_coordinates)

        if num_modes is None:
            num_modes = max(1, int(np.floor(num_nodes / 10) - 1))  # choose at least 1 mode
            logger.info(f'Choosing num_modes as {num_modes} based on the number of nodes {num_nodes}')
        assert num_modes <= num_nodes - 1
        if num_modes > num_nodes / 10:
            logger.warning(f'The number of modes {num_modes} should be less/equal than 0.1 x number of nodes (= {num_nodes}).')

        if d_vals is None:
            if damping is None:
                logger.debug('Performing undamped analysis.')
                d_vals = [0 for _ in range(num_nodes - 1)]
            elif isinstance(damping, float):
                d_vals = [0 for _ in range(num_nodes - 1)]
            else:
                d_vals = [0 for _ in range(num_nodes - 1)]
        elif damping is not None:
            logger.warning('Applying elementwise damping and global proportional at the same time. Not sure how ANSYS handles this.')
        else:
            # pre generated elementwise damping
            pass

        if sl_force_vals is None:
            sl_force_vals = [0 for _ in range(num_nodes - 1)]
        if eps_duff_vals is None:
            eps_duff_vals = [0 for _ in range(num_nodes - 1)]
        if hyst_vals is None:
            hyst_vals = [0 for _ in range(num_nodes - 1)]
#         if isinstance(d_max, (float, int)) or d_max is None:
#             d_max_vec = [None for _ in range(num_nodes)]
#             d_max_vec[-1] = d_max
#             d_max = d_max_vec

        assert len(k_vals) == num_nodes - 1
        assert len(masses) == num_nodes
        assert len(d_vals) == num_nodes - 1
        #assert len(d_max) == num_nodes

        ansys = self.ansys

        ansys.prep7()
        # Nodes
        ntn_conns = [[] for i in range(num_nodes - 1)]
        for i, (node, x, y, z) in enumerate(nodes_coordinates):
            ansys.n(node, x, y, z)
            if i < num_nodes - 1 : ntn_conns[i] = [node]
            if i > 0: ntn_conns[i - 1].append(node)

        # Elements
        lastm = None
        for (node, x, y, z), m in zip(nodes_coordinates, masses):
        #     generate mass for every node
            # print(node)
            if m != lastm:
                mass = self.mass(m)

            ansys.type(mass[0])
            ansys.real(mass[1])
            ansys.e(node)
            lastm = m

        # build equivalent linear model first
        last_params = [None for _ in range(5)]

        for params in zip(ntn_conns, k_vals, d_vals, sl_force_vals, eps_duff_vals, hyst_vals):
            #      0, 1, 2,   3,      4,    5
            ntn_conn, k, d, fsl, eps_duff, hyst = params
            # print(ntn_conn)
            if fsl:

                logger.warning("friction untested")
                # fsl is provided, because the equation for its computation contains omega and d_mean, which is not known here
                if last_params[3] != fsl or last_params[1] != k or last_params[2] != d:
                    k_1 = k * 100  # factor could also be provided,
                    coulomb = self.coulomb(k_2=k, k_1=k_1, f_sl=fsl, d=d)
                ansys.type(coulomb[0])
                ansys.real(coulomb[1])
            else:
                if last_params[1] != k or last_params[2] != d:
                    voigt_kelvin = self.voigt_kelvin(k=k, d=d)
                ansys.type(voigt_kelvin[0])
                ansys.real(voigt_kelvin[1])

            ansys.e(*ntn_conn)
            if eps_duff != 0 or hyst!=0:
                logger.warning("nonlinear untested")
                #assert abs(nl_ity) <= 0.5
                # only the nonlinear part is added here, the linear part is in voigt-kelvin/coulomb

                #deltad = d_max[ntn_conn[0] - 1] + d_max[ntn_conn[1] - 1]  # assuming linear node-to-node connections and node numbering starting with 1

                if last_params[4] != eps_duff:
                    # increase k here by dividing because it was decreased before
                    # TODO: change definition of nonlinear, currently the stiffness is defined for the working point, which is rather impractical, e.g. because of this division
                    nonlinear = None#self.nonlinear(nl_ity, deltad, k / (1 - nl_ity))
                    raise NotImplementedError('Duffing Oscillator not yet implemented')
                ansys.type(nonlinear[0])
                ansys.real(nonlinear[1])
                ansys.e(*ntn_conn)

            last_params = params

        # boundary conditions
        # Specifies that DOF constraint values are to be accumulated.
        # Needed to restore constraints after forced displacement analyses
        # ansys.dcum(oper='ADD') # does not work

        ansys.d(node=nodes_coordinates[0][0], value=0, lab='UX', lab2='UY', lab3='UZ')  # ,lab4='RX', lab5='RY',lab6='RZ')
        
        # move constrained node to the end to ensure consistency with ANSYS node indexing -> bull shit
        #nodes_coordinates.append(nodes_coordinates.pop(0))
        
        for node, x, y, z in nodes_coordinates:
            ansys.d(node=node, value=0, lab='UX', lab2='UY')
        ansys.finish()

        self.nodes_coordinates = nodes_coordinates
        self.ntn_conns = ntn_conns
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        self.d_vals = d_vals
        self.k_vals = k_vals
        self.masses = masses
        self.eps_duff_vals = eps_duff_vals
        self.sl_force_vals = sl_force_vals
        self.hyst_vals = hyst_vals

        # TODO: Account for multiple coloumbs, to get frequency estimates with loose sliders
        # TODO: Account for nonlinear stiffness, i.e. operating point
        frequencies, _, _ = self.modal(damped=False)

        # undamped analysis
        if damping is None or damping == 0:
            alpha = None
            beta = None
            globdamp = False
            damped = False
        # constant modal damping approximated by global rayleigh proportional damping
        elif isinstance(damping, float) and num_modes >= 2:
            omega1 = frequencies[0] * 2 * np.pi
            omega2 = frequencies[-1] * 2 * np.pi
            zeta1 = damping
            zeta2 = damping
            alpha, beta = np.linalg.solve(np.array([[1 / omega1, omega1], [1 / omega2, omega2]]), [2 * zeta1, 2 * zeta2])
            logger.debug(f'Rayleigh parameters: {alpha}, {beta}')
            globdamp = True
            damped = True
        elif isinstance(damping, float) and num_modes == 1:
            omega = frequencies[0] * 2 * np.pi
            alpha = 0
            beta = 2 * damping / omega
            globdamp = False
            damped = True
        # variable modal damping  approximated by global rayleigh proportional damping
        elif len(damping) == 2 and not isinstance(damping[1], bool) and num_modes >= 2:
            omega1 = frequencies[0] * 2 * np.pi
            omega2 = frequencies[-1] * 2 * np.pi
            zeta1, zeta2 = damping
            alpha, beta = np.linalg.solve(np.array([[1 / omega1, omega1], [1 / omega2, omega2]]), [2 * zeta1, 2 * zeta2])
            logger.debug(f'Rayleigh parameters: {alpha}, {beta}')
            globdamp = True
            damped = True
        # mass/stiffness/rayleigh proportional damping either elementwise or global
        elif len(damping) == 3:
            # should already be incorporated into the model
            alpha, beta, globdamp = damping
            damped = True
        # constant structural damping via DMPSTR (modal, harmonic) or IWAN (transient) based on loss factor
        elif len(damping) == 2 and isinstance(damping[1], bool):
            raise RuntimeError('Structural/Hysteretic damping not implemented yet.')
        else:
            raise RuntimeError(f'An unsupported damping argument was specified: {damping}.')

        # global proportional damping
        if damped:
            if globdamp:
                self.ansys.prep7()
                self.ansys.alphad(alpha)
                self.ansys.betad(beta)
                self.ansys.finish()
            else:
                # if only zeta or alpha,beta was provided, but local damping required
                # check/ compute the d_vals
                # d_vals is a list and its length should fit
                # elementwise Rayleigh with (alpha,) beta
                logger.debug('Using elementwise damping.')
                if alpha > 1e-15:
                    logger.warning(f'Elementwise proportional damping takes only a stiffness factor {beta}. Neglecting mass factor {alpha}!')
                for i in range(num_nodes - 1):
                    d_val = k_vals[i] * beta
                    if d_vals[i] and d_vals[i] != d_val:
                        raise RuntimeError("You should not provide global and local damping together")
#                         print(f"Overwriting viscous dashpot value {d_vals[i]} for element {i} with {d_val}")
                    d_vals[i] = d_val

                lastk = None
                lastd = None
                self.ansys.prep7()
                i = 0
                for ntn_conn, k, d in zip(ntn_conns, k_vals, d_vals):
#                     print(ntn_conn)
                    # TODO: This is likely to fail, if non-proportional d_vals are provided
                    if k != lastk and d != lastd:
                        nset = self.voigt_kelvin_elements[i][1]
                        ansys.rmodif(nset, 2, d)
                        lastk = k
                        lastd = d
                        i += 1

                self.ansys.finish()

        if meas_nodes is not None:
            meas_nodes = np.sort(meas_nodes)
            logger.debug(f'Using node components {meas_nodes}')
            ansys.nsel(type='NONE')
#             if 1 not in meas_nodes:
#                 print('Warning: make sure to include the first (constrained) node in meas_nodes for visualization.')
            for meas_node in meas_nodes:
                for node, x, y, z in self.nodes_coordinates:
                    if node == meas_node:
                        break
                else:
                    raise RuntimeError(f'Meas node {meas_node} was not found in nodes_coordinates')
                # print(node)
                ansys.nsel(type='A', item='NODE', vmin=meas_node, vmax=meas_node)  # select only nodes of interest
            # ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
            # ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
        else:
            ansys.nsel(type='ALL')
            meas_nodes = [node for node, _, _, _ in nodes_coordinates]
            meas_nodes = np.sort(meas_nodes)

        ansys.cm(cname='meas_nodes', entity='NODE')  # and group into component assembly

        ansys.nsel(type='ALL')
        # print(meas_nodes)

        ansys.finish()
        # time.sleep(20)
        self.damping = damping
        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.globdamp = globdamp
        self.meas_nodes = np.array(meas_nodes)
        
        self.state[0] = True
        for i in range(1, len(self.state)):
            self.state[i] = False
        #print(self.state)
#         self.ntn_conns  =ntn_conns
#         self.fric_rats  =fric_rats
#         self.nl_ity_rats=nl_ity_rats
#         self.d_max      =d_max
#         self.f_scale    =f_scale

    def example_rod(self, num_nodes, damping=None, nl_stiff=None, sl_forces=None, freq_scale=1, num_modes=None,  # structural parameters
                    num_meas_nodes=None, meas_nodes=None,  # signal parameters
                    ** kwargs):
        '''
        argument:
            damping = None -> no damping
                      zeta -> global rayleigh,
                      (zeta, zeta) -> global rayleigh
                      (alpha,beta,True/False) -> global/elementwise proportional
                      (eta, True/False) -> Structural/Hysteretic (global -> only damped modal and harmonic analyses not TRANSIENT! / elementwise -> only transient (IWAN))
                      (Friction -> only local via sl_forces
                      IWAN needs an equivalent damping ratio and a stress-strain relationship (G,E?)
                        from which a monotonic strain-softening backbone curve \tau vs \gamma (considering shear) according to Masing's rule can be computed (Kausel Eq. 7.307),
                        in turn this can be approximated/discretized by N spring-slider systems in parallel (or a nonlinear stiffness
                        TRY IN SDOF first
    
        TODO:
        - nonlinear-equivalent
            - stiffness (as in SDOF systems), estimation of d_max
            - friction (equivalent per node and mode, averaging?)
            - structural damping (ANSYS routines, IWAN)
        '''
        
        
        ansys = self.ansys
        
        try:
            ansys.finish()
        except pyansys.errors.MapdlExitedError as e:
            print(e)
            ansys = self.start_ansys()
        ansys.clear()
        
        # Example structure Burscheid Longitudinal modes
        total_height = 200
        E = 2.1e11 * freq_scale ** 2  # for scaling the frequency band linearly; analytical solution is proportional to sqrt(E); (ratios are fixed for a rod 1:2:3:4:... or similar)
        # I=0.01416
        A = 0.0343
        rho = 7850
    
        num_nodes = int(num_nodes)
    
        section_length = total_height / (num_nodes - 1)
        nodes_coordinates = []
        k_vals = [0 for _ in range(num_nodes - 1)]
        masses = [0 for _ in range(num_nodes)]
        d_vals = None  # [0 for i in range(num_nodes-1)]
        eps_duff_vals = [0 for _ in range(num_nodes - 1)]
        sl_force_vals = [0 for _ in range(num_nodes - 1)]
        hyst_vals = [0 for _ in range(num_nodes - 1)]
        
        if isinstance(damping, (list, tuple)):
            if len(damping) == 2 and isinstance(damping[1], bool):
                hyst_damp = damping[0]
        else:
            hyst_damp = None
        
        for i in range(num_nodes):
            #  nodes_coordinates.append([i+1,0,0,section_length*i])
            nodes_coordinates.append([i + 1, 0, 0, 0])  # to disable Warning "Nodes are not coincident"
            masses[i] += 0.5 * rho * A * section_length
            if i >= 2:
                masses[i - 1] += 0.5 * rho * A * section_length
            if i >= 1:
                k_vals[i - 1] = E * A / section_length
                
                if nl_stiff is not None:
                    eps_duff_vals[i - 1] = nl_stiff
                if sl_forces is not None:
                    sl_force_vals[i - 1] = sl_forces
                if hyst_damp is not None:
                    hyst_vals[i - 1] = hyst_damp
        if num_meas_nodes is None and meas_nodes is None:
            meas_nodes = [node for node, _, _, _ in nodes_coordinates[1:]]  # exclude the constrained node
        elif num_meas_nodes is not None and meas_nodes is None:
            # step = int(np.floor(num_nodes/num_meas_nodes))
            # meas_nodes = np.concatenate(([1],np.arange(step,num_nodes+1,step)))
            meas_nodes = np.rint(np.linspace(2, num_nodes, int(num_meas_nodes))).astype(int)
            if len(meas_nodes) != (num_meas_nodes):
                logger.warning(f'Warning number of meas_nodes generated {len(meas_nodes)} differs from number of meas_nodes specified {num_meas_nodes}')
        elif num_meas_nodes is not None and meas_nodes is not None:
            raise RuntimeError(f'You cannot provide meas_nodes {meas_nodes} and num_meas_nodes {num_meas_nodes} at the same time')
        
        logger.info(f'Building an example rod structure with {num_nodes} nodes, {damping} % damping, considering the output of {num_modes} modes at {num_meas_nodes} measurement nodes.')
        
        self.build_mdof(nodes_coordinates=nodes_coordinates,
                        k_vals=k_vals, masses=masses, d_vals=d_vals, damping=damping,
                        sl_force_vals=sl_force_vals, eps_duff_vals=eps_duff_vals, hyst_vals=hyst_vals,
                        meas_nodes=meas_nodes, num_modes=num_modes,)
            
        return



    # def build_sdof(self, f_init=None, d_init=None, mass=None, **kwargs):
        # ansys = self.ansys
        # ansys.prep7()
        # # Nodes
        # ansys.n(20, 0, 0, 2)
        # ansys.n(10, 0, 0, 2)
        #
        # # Elements
        #
        # # mass
        #
        # if mass is not None:
            # ansys.type(mass[0])
            # ansys.real(mass[1])
            # ansys.e(20)
            #
        # # ansys.e(30)
        # # ansys.e(40)
        #
        # coulomb = kwargs.pop('coulomb', None)
        # voigt_kelvin = kwargs.pop('voigt_kelvin', None)
        # nonlinear = kwargs.pop('nonlinear', None)
        #
        # if coulomb is not None:
            # ansys.type(coulomb[0])
            # ansys.real(coulomb[1])
            # ansys.e(10, 20)
            #
        # if voigt_kelvin is not None:
            # ansys.type(voigt_kelvin[0])
            # ansys.real(voigt_kelvin[1])
            # ansys.e(10, 20)
            #
        # if nonlinear is not None:
            # ansys.type(nonlinear[0])
            # ansys.real(nonlinear[1])
            # ansys.e(10, 20)
        # # ansys.e(20,30)
        # # ansys.e(30,40)
        #
        # # boundary conditions
        # ansys.d(node=10, value=0, lab='UX', lab2='UY', lab3='UZ')
        # ansys.d(node=20, value=0, lab='UX', lab2='UY')
        # if f_init is not None:
            # ansys.f(node=20, lab='FZ', value=f_init)
        # if d_init is not None:
            # ansys.ic(node=20, lab='UZ', value=d_init)
        # # ansys.d(node=30,value=0,lab='UX',lab2='UY')
        # # ansys.d(node=40,value=0,lab='UX',lab2='UY')
        #
        # ansys.finish()

    def free_decay(self, d0, dscale=1, deltat=None, dt_fact=None, timesteps=None, num_cycles=None, **kwargs):
        '''
        to take d0 parameter, do all the initial displacement things and call transient with passing kwargs
        save last analysis run as a class variable: d0, time histories, signal and analysis parameters


        d0 = initial displacement
            'static' : displacement obtained from static analysis with unit top displacement
            'modes' : arithmetic mean of all mode shapes
            number > 0, mode_index of which to excite
        
        TODO: extend 3D
        '''
        
        num_nodes = self.num_nodes
        uz = 2

        if d0 == 'static':
            # static analysis for initial conditions (free decay)
    #         disp = mech.static(fz=np.array([[num_nodes//2, -100000000]]))
            disp = self.static(uz=np.array([[num_nodes, 1]]))
            # print(disp[:,:], disp.shape)
            mode_index = None
    #         disp = mech.static(fz=np.array([[num_nodes//2, -100000000]]), uz=np.array([[num_nodes, 1]]))
        elif d0 == 'modes':
            _, _, modeshapes = self.modal(damped=False)
            for mode in range(self.num_modes):
                modeshapes[:, :, mode] /= modeshapes[-1, uz, mode]
            mode_index = None
        elif isinstance(d0, int):
            _, _, modeshapes = self.modal(damped=False)
            for mode in range(self.num_modes):
                modeshapes[:, :, mode] /= modeshapes[-1, uz, mode]
            mode_index = d0
        else:
            raise RuntimeError(f"No initial displacement was specified d0: {d0}.")

        dt_fact, deltat, num_cycles, timesteps, mode_index = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps, mode_index)

        frequencies, damping, _ = self.modal(damped=True)

        logger.info(f"Generating free decay response with f {frequencies[mode_index]:1.3f}, zeta {damping[mode_index]:1.3f}, free_decay mode {mode_index+1}")

        d = np.full((timesteps, num_nodes), np.nan)
        if d0 == 'static':
            d[0, :] = disp[:, uz]
        elif d0 == 'modes':
            d[0, :] = np.mean(modeshapes[:, uz, :], axis=1)
        elif isinstance(d0, int):
            d[0, :] = modeshapes[:, uz, d0]
#             y0=modeshapes[self.meas_nodes-1,uz,d0]
#             omega_d = frequencies[d0]*2*np.pi
#             zeta=damping[d0]
#             print(f'initial acceleration {-omega_d**2*y0}')
#             print(f'initial velocity {omega_d/np.sqrt(1-zeta**2)*zeta*y0}')

        parameter_set = kwargs.pop('parameter_set', None)

        time_values, response_time_history = self.transient(d=d, deltat=deltat, timesteps=timesteps, parameter_set=parameter_set, **kwargs)

        self.decay_mode = d0
        self.t_vals_decay = time_values
        self.resp_hist_decay = response_time_history
        
        self.state[1] = True
        
        return time_values, response_time_history

    def ambient(self, f_scale, deltat=None, dt_fact=None, timesteps=None, num_cycles=None, seed=None, **kwargs):
        '''to take f_scale or more advanced parameters and call transient with passing kwargs
        save last analysis run as a class variable: f_scale, time histories, input time histories, signal and analysis parameters
        '''
        num_nodes = self.num_nodes

        dt_fact, deltat, num_cycles, timesteps, _ = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps)

        logger.info(f"Generating ambient response in frequency range [0 ... {self.frequencies[-1]:1.3f}], zeta range [{np.min(self.modal_damping):1.3f}-{np.max(self.modal_damping):1.3f}], number of modes {self.num_modes}")

        logger.warning("WARNING: Gaussian Noise Generator has not been verified!!!!!!")
        rng = np.random.default_rng(seed)
        
        #f = rng.normal(0, f_scale, (timesteps, num_nodes))
        
        phase = rng.uniform(-np.pi, np.pi, (timesteps // 2 + 1, num_nodes))
        Pomega = f_scale * np.ones_like(phase) * np.exp(1j * phase)
        f = np.empty((timesteps, num_nodes))
        for channel in range(num_nodes):
            f[:, channel] = np.fft.irfft(Pomega[:, channel], timesteps)
        
        # TODO: implement correlated random processes here
        # TODO: implement random impulses here
        try:
            time_values, response_time_history = self.transient(f=f, deltat=deltat, timesteps=timesteps, **kwargs)
        except Exception as e:
            err_file = os.path.join(self.ansys.directory, self.ansys.jobname + '.err')
            with open(err_file) as f:
                lines_found = []
                block_counter = -1
                while len(lines_found) < 10:
                    try:
                        f.seek(block_counter * 4098, os.SEEK_END)
                    except IOError:
                        f.seek(0)
                        lines_found = f.readlines()
                        break
                    lines_found = f.readlines()
                    block_counter -= 1
                logger.error(str(lines_found[-10:]))

                raise e

        self.inp_hist_amb = f
        self.t_vals_amb = time_values
        self.resp_hist_amb = response_time_history

        self.state[2] = True
        
        return time_values, response_time_history, f

    def impulse_response(self, impulses=[[], [], [], []], form='sine', mode='combined', deltat=None, dt_fact=None, timesteps=None, num_cycles=None, out_quant=['d'], **kwargs):
        '''
        impulses = list of list containing
            [ref_nodes, imp_forces, imp_times, imp_durs/-num_modes]
        if imp_dur is negative, it is interpreted, as the highest mode, which is excited by half of the wavepower 1/sqrt(2)  (-3dB) (only for half-sine)
        
        form = 'sine', 'step', 'rect'
        
        mode='matrix' return the IRF matrix for each reference node
        mode='combined' apply all impulses at the same time
        
        
        save last analysis run as a class variable: ref_nodes, IRF matrix, signal and analysis parameters
        
        
        Unit impulse -> IRF
        Rectangular impulse -> define impulse length (includes stepped load for infinite length)
        Half-Sine impulse -> define f_max and power at f_max

        '''

        def analytical_constants_half_sine(p0, k, omegan, zeta, tp):
            omegad = omegan * np.sqrt(1 - zeta ** 2)
            omegaf = np.pi / tp

#             print(f'tp {tp},\t Tn {2*np.pi/omegan},\t p0 {p0},\t k {k},\t zeta {zeta},\t omegaf {omegaf},\t omegan {omegan},\t  omegad {omegad}')

            if np.isclose(zeta, 0) and np.isclose(omegaf, omegad):
                raise NotImplementedError('Modal decomposition of impulses at resonance for undamped system is not implemented yet')

            ydyn = np.sqrt(p0 / k / ((1 - (omegaf / omegan) ** 2) ** 2 + (2 * zeta * omegaf / omegan) ** 2))
            A = ydyn ** 2 * 2 * zeta * omegaf / omegan
            B = -ydyn ** 2 * omegaf / omegad * (1 - (omegaf / omegan) ** 2 - 2 * zeta ** 2)
            C = ydyn ** 2 * (1 - (omegaf / omegan) ** 2)
            D = -ydyn ** 2 * 2 * zeta * omegaf / omegan
            E = np.exp(-omegan * zeta * tp) * (A * np.cos(omegad * tp) + B * np.sin(omegad * tp) - np.exp(omegan * zeta * tp) * D)
            F = np.exp(-omegan * zeta * tp) * (-A * np.sin(omegad * tp) + B * np.cos(omegad * tp)) - C * omegaf / omegad - D * zeta * omegan / omegad

            return A, B, C, D, E, F

        def analytical_response_half_sine(t_vals, omegan, zeta, A, B, C, D, E, F, tp, ts=0):

            omegad = omegan * np.sqrt(1 - zeta ** 2)
            omegaf = np.pi / tp

            imp_resp_ex = np.zeros_like(t_vals)

            this_inds = np.logical_and(t_vals > ts, t_vals < ts + tp)
            forced_resp = np.exp(-omegan * zeta * (t_vals[this_inds] - ts))\
                         * (A * np.cos(omegad * (t_vals[this_inds] - ts))
                            +B * np.sin(omegad * (t_vals[this_inds] - ts)))\
                         +C * np.sin(omegaf * (t_vals[this_inds] - ts))\
                         +D * np.cos(omegaf * (t_vals[this_inds] - ts))
            imp_resp_ex[this_inds] = forced_resp

            this_inds = t_vals >= ts + tp
            free_response = np.exp(-omegan * zeta * (t_vals[this_inds] - ts - tp))\
                             * (E * np.cos(omegad * (t_vals[this_inds] - ts - tp))
                                +F * np.sin(omegad * (t_vals[this_inds] - ts - tp)))

            imp_resp_ex[this_inds] = free_response

            return imp_resp_ex

        def work_integrand_half_sine(t, omegan, zeta, A, B, C, D, p_i, tp, ts=0):

            omegad = omegan * np.sqrt(1 - zeta ** 2)
            omegaf = np.pi / tp

            assert t > ts  and t < ts + tp

            forced_resp = np.exp(-omegan * zeta * (t - ts))\
                         * (A * np.cos(omegad * (t - ts))
                            +B * np.sin(omegad * (t - ts)))\
                         +C * np.sin(omegaf * (t - ts))\
                         +D * np.cos(omegaf * (t - ts))
            impulse = -np.cos(omegaf * (t - ts)) * p_i * omegaf

            return forced_resp * impulse

        def generate_forces(impulses, form, frequencies):

            forces = []
            imp_durs = []
            for impulse, (node, imp_force, imp_time, imp_dur) in enumerate(zip(*impulses)):

                f = np.zeros((timesteps, num_nodes))
                if form == 'step':
                    if imp_dur is not None:
                        logger.warning(f'Stepped impulses have infinite length. A finite length tp_i of {imp_dur} was provided. Setting tp_i to {t_total-imp_time}')
                    imp_dur = t_total - imp_time
                elif imp_dur < 0:  # mode number, which should be excited with at least half energy
                    # it must be an integer and less than the total number of modes (it is an index starting at 0)
                    omega = frequencies[-imp_dur] * 2 * np.pi
                    if form == 'sine':
                        imp_dur = 3.72916 / omega
                    elif form == 'rect':
                        imp_dur = 2.78311 / omega
                imp_durs.append(imp_dur)
                assert imp_time + imp_dur <= t_total
                if imp_time < deltat:
                    raise ValueError(f'Starting time {imp_time} at node {node} must be at least deltat {deltat}')
                step_s = int(imp_time / deltat) - 1  # effectively rounds down
                step_e = int((imp_time + imp_dur) / deltat - 1)

                if form == 'step':
                    f[step_s:, node - 1] = imp_force
                elif form == 'rect':  # Stepped or Rectangular
                    f[step_s:step_e, node - 1] = imp_force
                elif form == 'sine':  # Half Sine
                    f[step_s:step_e, node - 1] = np.sin(np.linspace(0, np.pi, step_e - step_s, endpoint=False)) * imp_force
                else:
                    raise ValueError(f'Did not understand value of argument "form": {form}')
                forces.append(f)
            impulses[3] = imp_durs
            return forces

        num_nodes = self.num_nodes

        # get the system matrices for analytical solutions
        _, _, modeshapes = self.modal(use_cache=False)  # use_cache=False has to be done, to assemble the correct system matrices into the .full file
        full_path = os.path.join(self.ansys.directory, self.ansys.jobname + '.full')
        full = pyansys.read_binary(full_path)
        # TODO: Check, that Nodes and DOFS are in the same order as in mode shapes, should be, since both are returned sorted
        _, K, M = full.load_km(as_sparse=False, sort=False)
        K += np.triu(K, 1).T  # convert to full from upper triangular
        M += np.triu(M, 1).T  # convert to full from upper triangular

        dt_fact, deltat, num_cycles, timesteps, _ = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps)

        self.transient_parameters(**kwargs)

        frequencies, damping, _ = self.numerical_response_parameters()  # use numerical response parameters to ensure correct analytical responses

        num_modes = self.num_modes

        if mode == 'matrix':

            if form != 'step':
                logger.warning(f'You have chosen to generate an IRF matrix with other than stepped impulses, but {form}. Know what you are doing.')
            if len(out_quant) > 1:
                raise ValueError(f'For IRF Matrix mode, out_quant can only be a single quantity: {out_quant}')
            quant_ind = ['d', 'v', 'a'].index(out_quant[0])

            ref_nodes, imp_forces, imp_times, imp_durs = impulses
            for ref_node in ref_nodes:
                assert isinstance(ref_node, (int, np.int64))
            num_impulses = len(ref_nodes)
            num_ref_nodes = len(ref_nodes)
            num_meas_nodes = len(self.meas_nodes)

            IRF_matrix = np.zeros((num_ref_nodes, timesteps, num_meas_nodes, 3))
            F_matrix = np.zeros((num_ref_nodes, timesteps))
            energy_matrix = np.zeros((num_ref_nodes, num_modes))
            amplitudes_matrix = np.zeros((num_ref_nodes, num_modes))
            last_ref_node = 0
            for i in range(num_impulses):
                if imp_times[i] != deltat:
                    logger.warning(f'A starting time was specified as {imp_times[i]} which is invalid for IRF matrix mode. Starting time will be set to {deltat}.')
                this_ref_node = ref_nodes[i]
                if this_ref_node <= last_ref_node:
                    logger.warning(f'ref nodes should be in ascending order and not appear twice (last: {last_ref_node}, current: {this_ref_node}')
                last_ref_node = this_ref_node
                this_impulses = [(this_ref_node,), (imp_forces[i],), (deltat,), (imp_durs[1],)]
                time_values, response_time_history, f, energies, amplitudes = self.impulse_response(this_impulses, form, 'combined', deltat, dt_fact, timesteps, num_cycles, out_quant=out_quant, **kwargs)

                if form == 'step' and out_quant[0] == 'd':
                    disp = self.static(((this_ref_node, imp_forces[i]),), use_meas_nodes=True)
                    IRF_matrix[i, :, :, :] = (response_time_history[0] - disp)
                else:
                    IRF_matrix[i, :, :, :] = response_time_history[quant_ind]

                # in normal irf mode, impulses may be at all nodes, at any time
                # in irf matrix mode, impulses are only at ref node and
                # only at the beginning of the time series / signal
                # so we can use only the respective rows from the
                # input, energy and amplitudes matrix
                
                F_matrix[i, :] = f[:, this_ref_node - 1]
                energy_matrix[i, :] = energies[this_ref_node - 1, :]
                amplitudes_matrix[i, :] = amplitudes[this_ref_node - 1, :]
            
            # invalidate single signal impulse reponses
            self.inp_hist_imp = None
            self.resp_hist_imp = None
            self.modal_imp_energies = None
            self.modal_imp_amplitudes = None
            self.state[3] = False
            
            # save IRF matrix results
            self.t_vals_imp = time_values
            self.IRF_matrix = IRF_matrix
            self.imp_hist_imp_matrix = F_matrix
            self.modal_imp_energy_matrix = energy_matrix
            self.modal_imp_amplitude_matrix = amplitudes_matrix
            self.state[6] = True
            
            return time_values, IRF_matrix, F_matrix, energy_matrix, amplitudes_matrix

        logger.info(f"Generating impulse response in frequency range [0 ... {self.frequencies[-1]:1.3f}], zeta range [{np.min(self.modal_damping):1.3f}-{np.max(self.modal_damping):1.3f}], number of modes {num_modes}")
        t_total = timesteps * deltat

#         nrows = int(np.ceil(np.sqrt(num_meas_nodes)))
#         ncols=int(np.ceil(num_meas_nodes/nrows))
#         fign,axesn = plot.subplots(nrows,ncols,squeeze=False, sharex=True, sharey=True)
#         axesn=axesn.flatten()

        forces = generate_forces(impulses, form, frequencies)
        f = np.sum(forces, axis=0)  # sum up over the list of forces, which are all in the shape (timesteps, num_nodes)
        assert f.shape == (timesteps, num_nodes)

        time_values, response_time_history = self.transient(f=f, deltat=deltat, timesteps=timesteps, out_quant=out_quant, **kwargs)

#         t_vals_fine=np.linspace(0,t_total, timesteps*10)

        # This merges all impulses at the same node into a single line in the array
#         modal_imp_responses = np.zeros((t_vals_fine.size, num_nodes, num_modes))
        modal_imp_energies = np.zeros((num_nodes, num_modes))
        modal_amplitudes = np.zeros((num_nodes, num_modes))

        for i, (node_i, p_i, ts_i, tp_i) in enumerate(zip(*impulses)):

            if node_i in self.meas_nodes:
                nod_ind = np.where(self.meas_nodes == node_i)[0][0]
            else:
                nod_ind = None

#             nrows = int(np.ceil(np.sqrt(num_modes)))
#             ncols=int(np.ceil(num_modes/nrows))
#             figm,axesm = plot.subplots(nrows,ncols,squeeze=False, sharex=True, sharey=True)
#             axesm=axesm.flatten()

            omegaf_i = np.pi / tp_i

            f_i = forces[i][:, node_i - 1]

#             if form=='sine':# Half Sine
#                 t_vals_int = np.linspace(0,tp_i,10*int(tp_i/deltat))
#                 imp_int = -np.cos(omegaf_i*t_vals_int)*p_i*omegaf_i # negative of first derivative, so we don't have to compute the velocities
#             else:
            t_vals_int = np.linspace(0, tp_i, 2)  # we only need start and end values

#             sum_energies=0

            for mode in range(self.num_modes):

                phi_ij = modeshapes[node_i - 1, 2, mode]

                modeshape_j = modeshapes[:, :, mode].flatten()
                kappa_j = np.real(modeshape_j.T.dot(K).dot(modeshape_j))

                zeta_j = damping[mode]
                omegan_j = frequencies[mode] * 2 * np.pi / np.sqrt(1 - zeta_j ** 2)
                if form == 'sine':
                    A, B, C, D, E, F = analytical_constants_half_sine(p_i * np.abs(phi_ij), kappa_j, omegan_j, zeta_j, tp_i)

                    modal_amplitudes[node_i - 1, mode] += np.sqrt(E ** 2 + F ** 2) * np.abs(phi_ij)

#                     modal_imp_responses[:,node_i-1,mode] += analytical_response_half_sine(t_vals_fine, omegan_j, zeta_j, A, B, C, D, E, F, tp_i, ts_i)*np.abs(phi_ij)

                    # compute work with x10 finer resolution than model
                    imp_resp_int = analytical_response_half_sine(t_vals_int, omegan_j, zeta_j, A, B, C, D, E, F, tp_i)
#                     if np.isclose(zeta_j,0) and np.isclose(omegaf_i,frequencies[mode]*2*np.pi):
#                         print('Warning: Untested')
#                         W=(p_i*np.abs(phi_ij))**2/kappa_j*np.pi**2/8
#                     if np.isclose(zeta_j,0):
#                         W=(p_i*np.abs(phi_ij))**2/kappa_j*(omegaf_i*omegan_j/(omegaf_i**2-omegan_j**2))**2*(np.cos(omegan_j*tp_i)+1)
                    
                    # Integration for half-sine impulse by quad integration
                    W = scipy.integrate.quad(work_integrand_half_sine, a=t_vals_int[0], b=t_vals_int[-1], args=(omegan_j, zeta_j, A, B, C, D, p_i * np.abs(phi_ij), tp_i))[0]

                elif form == 'step' or form == 'rect':

#                     this_t_vals = t_vals_fine-ts_i
#                     modal_imp_responses[this_t_vals>=0,node_i -1,mode] += np.abs(phi_ij)*p_i*np.abs(phi_ij)/kappa_j*(1-np.exp(-zeta_j*omegan_j*this_t_vals)*(np.cos(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)+zeta_j/np.sqrt(1-zeta_j**2)*np.sin(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)))[this_t_vals>=0]
#                     this_t_vals = t_vals_fine-ts_i-tp_i
#                     modal_imp_responses[this_t_vals>=0,node_i -1,mode] += -np.abs(phi_ij)*p_i*np.abs(phi_ij)/kappa_j*(1-np.exp(-zeta_j*omegan_j*this_t_vals)*(np.cos(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)+zeta_j/np.sqrt(1-zeta_j**2)*np.sin(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)))[this_t_vals>=0]

                    if form == 'rect':
                        modal_amplitudes[node_i - 1, mode] += p_i * np.abs(phi_ij) / kappa_j * np.sqrt((1 + zeta_j ** 2 / (1 - zeta_j ** 2)) * (2 - 2 * np.cos(omegan_j * np.sqrt(1 - zeta_j ** 2) * tp_i))) * np.abs(phi_ij)
                    elif form == 'step':
                        modal_amplitudes[node_i - 1, mode] += p_i * np.abs(phi_ij) / kappa_j * (np.sqrt(1 + zeta_j ** 2 / (1 - zeta_j ** 2))) * np.abs(phi_ij)

                    imp_resp_int = p_i * np.abs(phi_ij) / kappa_j * (1 - np.exp(-zeta_j * omegan_j * t_vals_int) * (np.cos(omegan_j * np.sqrt(1 - zeta_j ** 2) * t_vals_int) + zeta_j / np.sqrt(1 - zeta_j ** 2) * np.sin(omegan_j * np.sqrt(1 - zeta_j ** 2) * t_vals_int)))
                    W = p_i * np.abs(phi_ij) * (imp_resp_int[-1] - imp_resp_int[0])

#                 axesm[mode].plot(t_vals_fine, modal_imp_responses[:,node_i -1,mode]) # in physical coordinates
#                 axesm[mode].plot(time_values, np.abs(phi_ij)*f_i*np.abs(phi_ij)/kappa_j,ls='dashed') # static solution, modal static response to modal impulse transformed to physical static solution to modal impulse and scaled to physical impulse

#                 if form == 'step': shift=p_i*np.abs(phi_ij)**2/kappa_j
#                 else: shift=0
#
#                 axesm[mode].axhline(modal_amplitudes[node_i -1,mode]+shift, color='grey')
#                 axesm[mode].axhline(-modal_amplitudes[node_i -1,mode]+shift, color='grey')

                modal_imp_energies[node_i - 1, mode] += W
#                 sum_energies += W
#             figm.suptitle(f'Modal impulse responses at node: {node_i}')

#             if node_i in self.meas_nodes:
#                 nod_ind = np.where(self.meas_nodes==node_i)[0][0]
#                 W=scipy.integrate.simps(response_time_history[1][:,nod_ind]*f_i,time_values, even='first')
#                 print(f'Total impulse energy of impulse {i}: {W}. Sum of modal impulse energies: {sum_energies} (Should match approximately)')
#
# #                 for l, label in enumerate(['d']):#,'v','a']):
# #                     axesn[nod_ind].plot(time_values, response_time_history[l][:,nod_ind], label=label, ls='none', marker='+')
# #
# #                 for mode in range(self.num_modes):
# #                     axesn[nod_ind].plot(t_vals_fine, modal_imp_responses[:,node_i -1,mode], alpha=.75, label=f'{mode}')
# #                 axesn[nod_ind].plot(t_vals_fine-0.0*deltat, np.sum(modal_imp_responses[:,node_i -1,:],axis=1), alpha=.75, label='modal sum')
# #
# #                 disp = self.static(((node_i,p_i),))[node_i-1,2]
# #                 k_glob = p_i/disp
# #                 axesn[nod_ind].plot(time_values, f[:,node_i-1]/k_glob, ls='dashed')
# #                 axesn[nod_ind].legend()
#
#             else:
#                 nod_ind = None
#                 print(f'Response at node {node_i} not available (not in meas_nodes), only sum of modal energies can be computed: {sum_energies}')
#
#         for nod_ind, node_i in enumerate(self.meas_nodes):
#             W=scipy.integrate.simps(response_time_history[1][:,nod_ind]*f[:,node_i-1],time_values, even='first')
#             print(f'Total impulse energy at node {node_i}: {W}. Sum of energies of all impulses: {np.sum(modal_imp_energies[node_i -1,:])} (Should match approximately)')

        self.inp_hist_imp = f
        self.t_vals_imp = time_values
        self.resp_hist_imp = response_time_history
        self.modal_imp_energies = modal_imp_energies
        self.modal_imp_amplitudes = modal_amplitudes
        
        # TODO: reduce modal_matrices to size (num_meas_nodes, num_modes)
        
        self.state[3] = True

        return time_values, response_time_history, f , modal_imp_energies, modal_amplitudes

    def frequency_response(self, N, inp_node, dof, fmax=None, out_quant='a'):
        '''
        Returns the onesided FRF matrix of the linear(ized) system
        at N//2 + 1 frequency lines for all nodes in meas_nodes
        by default the accelerance with input force at the last node is returned
        
        Uses numerically computed modal parameters and discrete system matrices
        The FRF may not be completely equivalent to analytical solutions
        
        inp_node is the ANSYS node number -> index is corresponding to
            meas_nodes (if compensated) or
            nodes_coordinates if not compensated
        '''
        
        nodes_coordinates = self.nodes_coordinates
        
        for i, (node, x, y, z) in enumerate(nodes_coordinates):
            if node == inp_node:
                inp_node_ind = i
                break
        else:
            raise RuntimeError(f'input node {inp_node} could not be found in nodes_coordinates')
        
        dof_ind = ['ux', 'uy', 'uz'].index(dof)
        
        # too complicated to get compensated (numerical damping, period elongation) modal_matrices, so we will live with a marginal error
        _, _, mode_shapes, kappas, _, _ = self.modal(num_modes=10, modal_matrices=True)
        frequencies, damping, mode_shapes_n = self.numerical_response_parameters(compensate=True, dofs=[dof_ind])
        # we have to distinguish between input and output mode shapes
        # output mode shapes are generally only for meas_nodes,
        # while input mode shapes must be complete i.e. input node could not be in meas_nodes
        
        
        if fmax is None:
            fmax = np.max(frequencies)
        
        df = fmax / (N // 2 + 1)
    
        omegas = np.linspace(0, fmax, N // 2 + 1, False) * 2 * np.pi
        assert np.isclose(df * 2 * np.pi, (omegas[-1] - omegas[0]) / (N // 2 + 1 - 1))
        omegas = omegas[:, np.newaxis]
        
        num_modes = self.num_modes
        num_meas_nodes = len(self.meas_nodes)
        omegans = frequencies * 2 * np.pi
        
        frf = np.zeros((N // 2 + 1, num_meas_nodes), dtype=complex)
        
        for mode in range(num_modes):
            
            omegan = omegans[mode]
            zeta = damping[mode]
            kappa = kappas[mode]
            mode_shape = mode_shapes_n[:, mode]
            modal_coordinate = mode_shapes[inp_node_ind, dof_ind, mode]
            # TODO: extend 3D
            
            if out_quant == 'a':
                frf += -omegan**2 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
            elif out_quant == 'v':
                frf += omegan / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
            elif out_quant == 'd':
                frf += 1 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
            else:
                logger.warning(f'This output quantity is invalid: {out_quant}')
            
        return omegas, frf
    
    def ambient_ifrf(self, f_scale, deltat=None, dt_fact=None, timesteps=None, num_cycles=None, out_quant=['d', 'v', 'a'], dofs=['ux', 'uy', 'uz'], seed=None, **kwargs):
        '''
        a shortcut function for ambient using the linear frf and the
        inverse FFT to generate the signal by orders of magnitude faster
        non-linear effects, or non-white noise inputs are not possible
        '''
        num_nodes = self.num_nodes
        
        dt_fact, deltat, num_cycles, timesteps, _ = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps)
        
        logger.info(f"Generating ambient response (IFFT/FRF-based) in frequency range [0 ... {self.frequencies[-1]:1.3f}], zeta range [{np.min(self.modal_damping):1.3f}-{np.max(self.modal_damping):1.3f}], number of modes {self.num_modes}")

        logger.warning("WARNING: Gaussian Noise Generator has not been verified!!!!!!")
        
        meas_nodes = self.meas_nodes
        input_nodes = [node for node, x, y, z in self.nodes_coordinates[-2:-1]]
        
        rng = np.random.default_rng(seed)
        phase = rng.uniform(-np.pi, np.pi, (timesteps // 2 + 1, num_nodes))
        Pomega = f_scale * np.ones_like(phase) * np.exp(1j * phase)
        
        response_time_history = [None, None, None]
        
        for quant_ind, quant in enumerate(['d', 'v', 'a']):
            if quant in out_quant:
                sig = np.zeros((timesteps, len(meas_nodes), 3))
                # compute ifft for each combination of input and output node
                # use linear superposition of output signals from each input node
                
                for inp_node_ind, inp_node in enumerate(input_nodes):
                    for dof_ind, dof in enumerate(['ux', 'uy', 'uz']):  # use all dofs in order to comply with the output of a transient simulation
                        if dof in dofs:
                            _, this_frf = self.frequency_response(N=timesteps, inp_node=inp_node, dof=dof,
                                                                  fmax=1 / deltat / 2, out_quant=quant, )
                            for channel in range(this_frf.shape[1]):
                                sig[:, channel, dof_ind] += np.fft.irfft(this_frf[:, channel] * Pomega[:, inp_node_ind])
                response_time_history[quant_ind] = sig
        
        time_values = np.linspace(deltat, timesteps * deltat, timesteps) #  ansys also starts at deltat
        
        self.inp_hist_amb = None
        self.t_vals_amb = time_values
        self.resp_hist_amb = response_time_history

        self.state[2] = True
        
        return time_values, response_time_history, None
    
    def static(self, fz=None, uz=None, use_meas_nodes=False):
        '''
        provide fz/uz as a numpy array of [node, displacement] pairs
        assumes UZ DOF, may have to be changed in the future
        constraints are restored after forced-displacement analysis, currently hardcoded
        
        TODO: extend 3D
        '''

        if fz is not None:
            if not isinstance(fz, np.ndarray):
                fz = np.array(fz)
            assert fz.shape[1] == 2

        if uz is not None:
            assert isinstance(uz, np.ndarray)
            assert uz.shape[1] == 2

        ansys = self.ansys

        # Static Response
        ansys.slashsolu()
        ansys.antype('STATIC')

        ansys.nsel(type='ALL')

        if fz is not None:
            for node, fz_ in fz:
                ansys.f(node=node, lab='FZ', value=fz_)

        if uz is not None:
            for node, uz_ in uz:
                ansys.d(node=node, lab='UZ', value=uz_)

        ansys.outres(item='ERASE')

        if use_meas_nodes:
            ansys.outres(item='ALL', freq='NONE')
            ansys.outres(item='NSOL', freq='LAST', cname='MEAS_NODES')

        ansys.solve()

        if fz is not None:
            ansys.fdele(node='ALL', lab='ALL')

        if uz is not None:
            # remove all prescribed displacements including constraints
            ansys.ddele(node='ALL', lab='ALL')
            # restore constraints
            ansys.d(node=self.nodes_coordinates[0][0], value=0, lab='UX', lab2='UY', lab3='UZ')  # ,lab4='RX', lab5='RY',lab6='RZ')
            for node, x, y, z in self.nodes_coordinates:
                ansys.d(node=node, value=0, lab='UX', lab2='UY')

        ansys.finish()

        #self.last_analysis = 'static'
        res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + '.rst'))

        nodes, disp = res.nodal_solution(0)
#         print(nodes,disp)

        return disp

    def modal(self, damped=True, num_modes=None, use_cache=True, reset_sliders=True, modal_matrices=False):  # Modal Analysis
        ansys = self.ansys

        num_nodes = self.num_nodes
        if num_modes is None:
            num_modes = self.num_modes
        assert num_modes <= num_nodes
        if num_modes > 10 * num_nodes:
            logger.warning(f'The number of modes {num_modes} should be greater/equal than 10 number of nodes {num_nodes}.')

        # cached modal analysis results
        # TODO: the logic needs improvement: num_modes may have been different for both types of analyses
        if damped and num_modes == self.num_modes and use_cache and not modal_matrices:
            if self.damped_frequencies is not None:
                frequencies = self.damped_frequencies
                damping = self.modal_damping
                mode_shapes = self.damped_mode_shapes
                return frequencies, damping, mode_shapes
        elif not damped and num_modes == self.num_modes and use_cache and not modal_matrices:
            if self.frequencies is not None:
                frequencies = self.frequencies
                mode_shapes = self.mode_shapes
                damping = np.zeros_like(frequencies)
                return frequencies, damping, mode_shapes

        ansys.prep7()
        if self.coulomb_elements and reset_sliders:
            logger.info("Temporarily resetting sliders for modal analysis.")
            real_constants = []
            # reset the sliding capabilities (k1,fslide) for modal analysis,
            # else ktot=k2+k1 would be taken
            for coulomb in self.coulomb_elements:
                if coulomb is None: continue
                nset = coulomb[1]
                # print(ansys.rlist(nset))
                ansys.get('k_coul', entity="rcon", entnum=nset, item1=1,)  # r5=f_sl_in, r6=k_2
                ansys.get('f_coul', entity="rcon", entnum=nset, item1=5,)  # r5=f_sl_in, r6=k_2
                # ansys.load_parameters()
                real_constants.append((ansys.parameters["K_COUL"], ansys.parameters["F_COUL"]))
                # print(real_constants)
                ansys.rmodif(nset, 1, 0)
                ansys.rmodif(nset, 5, 0)
                # print(ansys.rlist(nset))
                # print(real_constants[-1])
        ansys.finish()

        ansys.run('/SOL')
        ansys.antype('MODAL')

        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')  # Disable all output
        # ansys.nsel(type='S', item='NODE', vmin=20,vmax=20) # select only nodes of interest
        # ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
        # ansys.nsel(type='ALL')

        ansys.nsel(type='ALL')
        ansys.outres(# item='A',
                     item='NSOL',
                     freq='ALL'
                    # ,cname='meas_nodes'# for modal matrices we need the full mode shapes
                     )  # Controls the solution data written to the database.
        if damped:
            ansys.modopt(method='QRDAMP', nmode=num_modes, freqb=0,
                         freqe=1e4,
                         cpxmod='cplx',
                         nrmkey='off',
                         )
#             ansys.modopt(method='DAMP',nmode=num_modes,freqb=0,
#                          freqe=1e8,
#                          cpxmod='cplx',
#                          nrmkey='on',
#                          )
        else:
            ansys.modopt(method='LANB', nmode=num_modes, freqb=0,
                         freqe=1e4,
                         nrmkey='off',
                         )

        ansys.mxpand(nmode='all', elcalc=1)
        ansys.solve()
        
        if modal_matrices:
            ansys.wrfull()
            ansys.finish()
            ansys.aux2()
            '''
            APDL Guide 4.4
            The mode shapes from the .MODE file and the DOF results from 
            the .RST file are in the internal ordering, and they need 
            to be converted before use with any of the matrices from the .FULL file, as shown below:
            '''
            ansys.smat(matrix='Nod2Solv', type='D', method='IMPORT', val1='FULL', val2=f"{self.jobname}.full", val3="NOD2SOLV")
            # Permutation operators can not be exported, so the mode shapes must be redordered to solver ordering in APDL 
            if damped:
                ansys.dmat(matrix="PhiI", type="Z", method="IMPORT", val1="MODE", val2=f"{self.jobname}.mode")
            else:
                ansys.dmat(matrix="PhiI", type="D", method="IMPORT", val1="MODE", val2=f"{self.jobname}.mode")
            ansys.mult(m1='Nod2Solv', t1='', m2='PhiI', t2='', m3='PhiB')
            ansys.export(matrix="PhiB", format="MMF", fname="PhiB.bin")
            msh = np.array(scipy.io.mmread('PhiB.bin'))
            
            ansys.dmat(matrix="MatKD", type="D", method="IMPORT", val1="FULL", val2=f"{self.jobname}.full", val3="STIFF")
            ansys.export(matrix="MatKD", format="MMF", fname="MatKD.bin")
            K = np.array(scipy.io.mmread('MatKD.bin'))
            ansys.dmat(matrix="MatMD", type="D", method="IMPORT", val1="FULL", val2=f"{self.jobname}.full", val3="MASS")
            ansys.export(matrix="MatMD", format="MMF", fname="MatMD.bin")
            M = scipy.io.mmread('MatMD.bin')
            try:
                ansys.dmat(matrix="MatCD", type="D", method="IMPORT", val1="FULL", val2=f"{self.jobname}.full", val3="DAMP")
                ansys.export(matrix="MatCD", format="MMF", fname="MatCD.bin")
                C = scipy.io.mmread('MatCD.bin')
            except Exception as e:
                # print(e)
                C = np.zeros_like(K)
                # compute modal matrices
            
            kappas = np.zeros((num_modes))
            mus = np.zeros((num_modes))
            etas = np.zeros((num_modes))
            
            for mode in range(num_modes):
                # TODO: should work, since I assume K, M and C are 3D
                # properly remove constraint nodes
                # check complex conjugate?
                msh_f = msh[:, mode]

                kappas[mode] = (msh_f.T.dot(K).dot(msh_f.conj())).real
                mus[mode] = (msh_f.T.dot(M).dot(msh_f.conj())).real
                etas[mode] = (msh_f.T.dot(C).dot(msh_f.conj())).real
        
        
        ansys.finish()
        
        ansys.prep7()
        if self.coulomb_elements and reset_sliders:
            for coulomb, real_constant in zip(self.coulomb_elements, real_constants):
                nset = coulomb[1]
                ansys.rmodif(nset, 1, real_constant[0])
                ansys.rmodif(nset, 5, real_constant[1])
        ansys.finish()

        #self.last_analysis = 'modal'

        res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + '.rst'))

        num_modes_ = res.nsets
        if res._resultheader['cpxrst']:  # real and imaginary parts are saved as separate sets
            num_modes_ //= 2
        if num_modes_ != num_modes:
            logger.warning(f'The number of numerical modes {num_modes_} differs from the requested number of modes {num_modes}.')
            num_modes = num_modes_

        nnodes = res._resultheader['nnod']
        assert nnodes == self.num_nodes
        ndof = res._resultheader['numdof']

        mode_shapes = np.full((nnodes, ndof, num_modes), (1 + 1j) * np.nan, dtype=complex)
        frequencies = np.full(num_modes, np.nan)
        damping = np.full(num_modes, np.nan)

        # print(res.time_values)
        if res._resultheader['cpxrst']:
            for mode in range(num_modes):
                sigma = res.time_values[2 * mode]
                omega = res.time_values[2 * mode + 1]
                if omega < 0 : continue  # complex conjugate pair

                frequencies[mode] = omega  # damped frequency
                damping[mode] = -sigma / np.sqrt(sigma ** 2 + omega ** 2)

                mode_shapes[:, :, mode].real = res.nodal_solution(2 * mode)[1]
                mode_shapes[:, :, mode].imag = res.nodal_solution(2 * mode + 1)[1]
            else:
                nnum = res.nodal_solution(0)[0]

        else:
            frequencies[:] = res.time_values
            for mode in range(num_modes):
                nnum, modal_disp = res.nodal_solution(mode)
                mode_shapes[:, :, mode] = modal_disp
            mode_shapes = mode_shapes.real
            

        
        # reduce mode shapes to meas_nodes and translational dof
        # if self.meas_nodes is not None:
            # meas_indices = []
            # for meas_node in self.meas_nodes:
                # for i, (node, _, _, _) in enumerate(self.nodes_coordinates):
                    # if node == meas_node:
                        # meas_indices.append(i)
                        # break
                # else:
                    # raise RuntimeError(f'meas_node {meas_node} could not be found in nodes_coordinates')
            # mode_shapes = mode_shapes[meas_indices, :3, :]
        mode_shapes = mode_shapes[:, :3, :]
        
        if damped:
            self.damped_frequencies = frequencies
            self.modal_damping = damping
            self.damped_mode_shapes = mode_shapes
        else:
            self.frequencies = frequencies
            self.mode_shapes = mode_shapes
        self.num_modes = num_modes
        
        self.state[4] = True
        
        if modal_matrices:
            return frequencies, damping, mode_shapes, kappas, mus, etas
        else:
            return frequencies, damping, mode_shapes

    def transient_parameters(self, meth='NMK', parameter_set=None, **kwargs):

        delta = None  # gamma
        alpha = None  # beta
        alphaf = None  # alphaf
        alpham = None  # alpham

        if meth == 'NMK':
            tintopt = 'NMK'
            if isinstance(parameter_set, tuple):
                assert len(parameter_set) == 2
                delta, alpha = parameter_set
            elif isinstance(parameter_set, str):
                if parameter_set == 'AAM':  # Trapezoidal Rule -> Constant Acceleration
                    delta = 1 / 2
                    alpha = 1 / 4
                elif parameter_set == 'LAM':  # Original Newmark 1/6 -> Linear Acceleration
                    delta = 1 / 2
                    alpha = 1 / 6
                else:
                    raise RuntimeError(f"Could not understand parameter set {parameter_set}.")
            elif isinstance(parameter_set, (float, int)):
                rho_inf = parameter_set
                delta = (3 - rho_inf) / (2 * rho_inf + 2)
                alpha = 1 / ((rho_inf + 1) ** 2)
            elif parameter_set is None:
                delta = 1 / 2
                alpha = 1 / 4
            else:
                raise RuntimeError(f"Could not understand parameter set {parameter_set}.")

        elif meth in ['HHT', 'WBZ', 'G-alpha']:
            tintopt = 'HHT'
            if isinstance(parameter_set, (float, int)):
                rho_inf = parameter_set
            elif parameter_set is None:
                logger.info("Spectral radius was not supplied. Using 1")
                rho_inf = 1
            else:
                assert len(parameter_set) == 4
                delta, alpha, alphaf, alpham = parameter_set

            if meth == 'HHT':  # HHT-α Hilber-Hugh Taylor
                alphaf = np.round((1 - rho_inf) / (rho_inf + 1), 8)
                if alphaf > 0.5: raise RuntimeError("ANSYS won't take alphaf>0.5 in the HHT method")
                alpham = 0
                delta = np.round(1 / 2 + alphaf, 8)
                alpha = np.round((1 + alphaf) ** 2 / 4, 9)
            elif meth == 'WBZ':  # WBZ-α Wood-Bosak-Zienkiewicz
                alphaf = 0
                alpham = np.round((rho_inf - 1) / (rho_inf + 1), 8)
                delta = np.round(1 / 2 - alpham, 8)
                alpha = np.round((1 - alpham) ** 2 / 4, 8)
            elif meth == 'G-alpha':
                alpham = np.round((2 * rho_inf - 1) / (rho_inf + 1), 8)
                alphaf = np.round(rho_inf / (rho_inf + 1), 8)
                delta = np.round(1 / 2 - alpham + alphaf, 8)
                alpha = np.round(1 / 4 * (1 - alpham + alphaf) ** 2, 8)
        else:
            raise ValueError(f'Method provided could not be understood: {meth}')

        tintopt = 'HHT'

        logger.debug(f'Transient analysis method {meth} with parameter set {parameter_set} -> gamma {delta}, beta {alpha}, alpha_f {alphaf}, alpha_m {alpham}')

        self.trans_params = (delta, alpha, alphaf, alpham)

        return delta, alpha, alphaf, alpham, tintopt

    def signal_parameters(self, dt_fact=None, deltat=None, num_cycles=None, timesteps=None, mode_index=None):
        # Setup transient simulation parameters

        assert deltat is not None or dt_fact is not None
        assert timesteps is not None or num_cycles is not None

        frequencies, damping, _ = self.modal(damped=True)

        if mode_index is None:
            f_min = frequencies[0]
            f_max = frequencies[-1]
            mode_index = self.num_modes - 1
        else:
            f_min = frequencies[mode_index]
            f_max = frequencies[mode_index]

        if dt_fact is None:
            deltat = np.round(deltat,8) # round deltat: in transient stepsize*deltat = loadstep end time.  ansys computes stepsize back from loadsteptime and deltat with finite precision, thus inconsitencies occur
            dt_fact = deltat * f_max
        elif deltat is None:
            deltat = dt_fact / f_max
            deltat = np.round(deltat,8)

        if timesteps is None:
            # ensure timesteps is a multiple of 2 to avoid problems in any fft based processing
            timesteps = int(np.ceil(num_cycles / f_min / deltat) // 2) * 2
        elif num_cycles is None:
            if timesteps % 2:
                logging.warning(f'Timesteps is {timesteps}, which is not a multiple of 2. In order to avoid problems in FFT-based processing, setting it to {timesteps +1}')
                timesteps += 1
            num_cycles = int(np.floor(timesteps * f_min * deltat))

        Omega = deltat * frequencies * 2 * np.pi
#         print(Omega)
        if (Omega > 0.1).any():
            if self.trans_params is not None:
                #  compute the bifurcation limits for all modes
                gamma, beta, alpha_f, alpha_m = self.trans_params
                if alpha_f is None and alpha_m is  None:
                    Omega_bif = (-0.5 * damping * (-gamma + 0.5) + np.sqrt(damping ** 2 * (beta - 0.5 * gamma) - beta + 0.25 * (gamma + 0.5) ** 2)) / (0.25 * (gamma + 0.5) ** 2 - beta)
                    if (Omega > Omega_bif).any():
                        warnings.warn(f'Omega exceeds bifurcation limit for modes {np.where(Omega>Omega_bif)[0]+1}')
            # timestep, where decay is down to 1e-8
            else:
                t_crit = -1 * np.log(1e-8) / damping[-1] / f_max / 2 / np.pi
                if t_crit <= timesteps * deltat:
                    logger.warning(f'For conditionally stable integration, convergence issues may appear in free-decay mode due to numerical precision leaking into higher modes: {np.where((deltat*frequencies*2*np.pi)>0.1)[0]+1}.')

        self.deltat = deltat
        self.timesteps = timesteps
        logger.info(f"Signal parameters for upcoming transient: deltat {deltat:1.6f}, dt_fact for f_max {dt_fact:1.6f}, timesteps {timesteps}, num_cycles for f_min {num_cycles}")

        return dt_fact, deltat, num_cycles, timesteps, mode_index

    def transient(self, f=None, d=None, timint=1, deltat=None, timesteps=None, out_quant=['d', 'v', 'a'],
                  chunksize=10000, chunk_restart=False, **kwargs):
        ansys = self.ansys

        chunksize = int(chunksize)
        num_chunks = timesteps // chunksize
        if num_chunks > 5 and not chunk_restart:
            logger.debug(f"{num_chunks}>5 chunks will be computed, enabling chunk_restart")
            chunk_restart = True
#             prior_log_level = ansys._log.level
#             ansys.set_log_level('INFO')
            ansys.config("NRES", chunksize)
#             ansys.set_log_level(prior_log_level)

        delta, alpha, alphaf, alpham, tintopt = self.transient_parameters(**kwargs)

        if deltat is not None and timesteps is not None:
            # check user provided signal parameters
            _, deltat, _, timesteps, _ = self.signal_parameters(deltat=deltat, timesteps=timesteps)
        elif deltat is None and timesteps is not None:
            deltat = self.deltat
            # check user provided signal parameters
            _, deltat, _, timesteps, _ = self.signal_parameters(deltat=deltat, timesteps=timesteps)
        if timesteps is None and deltat is not None:
            timesteps = self.timesteps
            # check user provided signal parameters
            _, deltat, _, timesteps, _ = self.signal_parameters(deltat=deltat, timesteps=timesteps)
        else:
            deltat = self.deltat
            timesteps = self.timesteps

        ansys.slashsolu()
        ansys.nsel(type='ALL')
        ansys.antype('TRANS')
#         ansys.set_log_level("INFO")
        ansys.trnopt(method='FULL', tintopt=tintopt)  # bug: vaout should be tintopt

        if chunk_restart:
            ansys.rescontrol(action='DEFINE', ldstep='LAST', frequency='LAST')  # Controls file writing for multiframe restarts
        else:
            ansys.rescontrol(action='DEFINE', ldstep='NONE', frequency='NONE', maxfiles=-1)  # Controls file writing for multiframe restarts

        ansys.kbc(1)  # Specifies ramped or stepped loading within a load step.
        ansys.timint(timint)  # Turns on transient effects.

        ansys.tintp(alpha=alpha if alpha is not None else '',
                    delta=delta if delta is not None else '',
                    alphaf=alphaf if alphaf is not None else '',
                    alpham=alpham if alpham is not None else '',
                    oslm="", tol="", avsmooth='')

        ansys.autots('off')
        ansys.nsubst(1)

        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')  # Disable all output, must be here, else everything, not just meas_nodes will be written to db

        t_start = time.time()

        # according to ansys structural analysis guide to achieve zero initial velocity
        if d is not None:
            ansys.timint('off')
            for i, (node, x, y, z) in enumerate(self.nodes_coordinates):
                if not i: continue  # skip first node, to not change constraints
                ansys.d(node=node, lab='UZ', value=d[0, i])
            ansys.time(deltat / 2)
            ansys.nsubst(2)
            ansys.solve()
            ansys.nsubst(1)
            ansys.timint(timint)
            for i, (node, x, y, z) in enumerate(self.nodes_coordinates):
                if not i: continue  # skip first node
                ansys.ddele(node=node, lab='UZ')

        if 'd' in out_quant:
            ansys.outres(item='NSOL', freq='ALL', cname='MEAS_NODES')

        if 'a' in out_quant:
            ansys.outres(item='A', freq='ALL', cname='MEAS_NODES')

        if 'v' in out_quant:
            ansys.outres(item='V', freq='ALL', cname='MEAS_NODES')

#         ansys.set_log_level("ERROR")
        t_end = t_start
        t_start = time.time()
        logger.debug(f'setup  in {t_start-t_end} s')

        # make sure time series start at t=deltat, a previous solve was done at t=deltat/2, and a constant deltim would shift everything by deltat/2
        # solve for consistent accelerations
        if d is not None:
            ansys.time(deltat)
#             print(deltat)
            ansys.solve()

        for chunknum in range(timesteps // chunksize + 1):

            if (chunknum + 1) * chunksize <= timesteps:
                stepsize = chunksize
            else:
                stepsize = timesteps % chunksize
            if stepsize == 0:
                break
            # print(chunknum, timesteps//chunksize, timesteps%chunksize, stepsize, chunksize, timesteps, (chunknum+1)*chunksize)
            if f is not None:
                table = np.zeros((stepsize + 1, self.num_nodes + 1))
                table[1:, 0] = np.arange(chunknum * chunksize + 1, chunknum * chunksize + 1 + stepsize) * deltat
                table[0, 1:] = np.arange(1, self.num_nodes + 1)
                table[1:, 1:] = f[chunknum * chunksize:chunknum * chunksize + stepsize, :]

                np.savetxt(f'{self.jobname}.csv', table)

                with supress_logging(ansys):
                    ansys.starset(par='EXCITATION')

                ansys.dim(par='EXCITATION', type='TABLE', imax=stepsize, jmax=self.num_nodes, kmax="", var1='TIME', var2='NODE')
                ansys.tread(par='EXCITATION', fname=f'{self.jobname}', ext='csv')

                ansys.f(node='ALL', lab='FZ', value='%EXCITATION%')

            # continue
#             ansys.set_log_level('INFO')
            if chunknum == 0:
                ansys.autots('off')
                # might become slightly inaccurate for "nonrational" deltats,
                # but using nsubst gives problems in free decay, because the first timestep was already computed outside the loop
                ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat)

            ansys.time((chunknum * chunksize + stepsize) * deltat)
            ansys.output(fname='null', loc='/dev')
            ansys.solve()
            ansys.output(fname='term')
            if chunk_restart:
                shutil.copyfile(os.path.join(ansys.directory, ansys.jobname + '.rst'), os.path.join(ansys.directory, ansys.jobname + f'.rst.{chunknum}'))
                ansys.get(par='RST_SUBSTEPS', entity='ACTIVE', item1='SOLU', it1num='NCMSS')
                rst_substeps = int(ansys.parameters['RST_SUBSTEPS'])
                ansys.finish()
                # not needed, if we just move the rst file, but ansys throws a warning about a missing rst file
                if False:
                    os.remove(os.path.join(ansys.directory, ansys.jobname + '.rst'))
                else:
                    ansys.aux3()
                    ansys.file(ansys.jobname, 'rst')
                    ansys.delete(set='SET', nstart=1, nend=rst_substeps + 1)
                    ansys.compress()
                    ansys.finish()
                # ansys.set_log_level('DEBUG')
                # ansys.config(lab='stat')
                ansys.slashsolu()
                ansys.antype(status='rest')  # restart last analysis
                # ansys.set_log_level("INFO")

                # outres is not restored upon restart
                if True:
                    ansys.outres(item='ERASE')
                    ansys.outres(item='ALL', freq='NONE')
                    if 'd' in out_quant:
                        ansys.outres(item='NSOL', freq='ALL', cname='MEAS_NODES')
                    if 'a' in out_quant:
                        ansys.outres(item='A', freq='ALL', cname='MEAS_NODES')
                    if 'v' in out_quant:
                        ansys.outres(item='V', freq='ALL', cname='MEAS_NODES')

            t_end = t_start
            t_start = time.time()
            logger.info(f'{stepsize} timesteps in {t_start-t_end:.3f} s')

#         else:
#             #TODO: align everything with the above procedure
#             if timesteps%chunksize:
#                 chunknum=timesteps//chunksize
#                 if f is not Nonem:
#                     table = np.zeros(((timesteps%chunksize)+1,self.num_nodes+1))
#                     table[1:,0]=np.arange(chunknum*chunksize+1,chunknum*chunksize+timesteps%chunksize+1)*deltat
#                     table[0,1:]=np.arange(1,self.num_nodes+1)
#                     table[1:,1:]=f[chunknum*chunksize:chunknum*chunksize+timesteps%chunksize,:]
#
#                     np.savetxt(f'{self.jobname}.csv',table)
#                     with supress_logging(ansys):
#                         ansys.starset(par='EXCITATION')
#                     ansys.dim(par='EXCITATION', type='TABLE', imax=timesteps%chunksize, jmax=self.num_nodes, kmax="",var1='TIME',var2='NODE')
#                     ansys.tread(par='EXCITATION', fname=f'{self.jobname}', ext='csv')
#
#                     ansys.f(node='ALL', lab='FZ', value='%EXCITATION%')
#
#                 ansys.autots('off')
#                 #ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat)
#                 ansys.time((timesteps)*deltat)
# #
#                 ansys.solve()
#                 if chunk_restart:
#                     os.rename(os.path.join(ansys.directory, ansys.jobname+'.rst'), os.path.join(ansys.directory, ansys.jobname+f'.rst.{chunknum}'))
#                 t_end=t_start
#                 t_start = time.time()
#                 logger.info(f'{timesteps%chunksize} timesteps in {t_start-t_end} s')

        ansys.set_log_level("WARNING")
        ansys.finish()

        #self.last_analysis = 'trans'

        if chunk_restart:
            out_a = []
            out_v = []
            out_d = []
            out_t = []

            for chunknum in range(timesteps // chunksize + int(bool(timesteps % chunksize))):
                res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + f'.rst.{chunknum}'))
                out_t.append(res.time_values)

                solution_data_info = res._solution_header(0)
                DOFS = solution_data_info['DOFS']
                ux = DOFS.index(1)
                uy = DOFS.index(2)
                uz = DOFS.index(3)

                if 'd' in out_quant:
                    out_d.append(res.nodal_time_history('NSL')[1])
                if 'a' in out_quant:
                    out_a.append(res.nodal_time_history('ACC')[1])
                if 'v' in out_quant:
                    out_v.append(res.nodal_time_history('VEL')[1])

            time_values = np.concatenate(out_t)

            if 'a' in out_quant:
                all_disp_a = np.concatenate(out_a, axis=0)[:, :, (ux, uy, uz)]
            else:
                all_disp_a = None
            if 'v' in out_quant:
                all_disp_v = np.concatenate(out_v, axis=0)[:, :, (ux, uy, uz)]
            else:
                all_disp_v = None
            if 'd' in out_quant:
                all_disp_d = np.concatenate(out_d, axis=0)[:, :, (ux, uy, uz)]
            else:
                all_disp_d = None

        else:
#             print("Reading binary")
            res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + '.rst'))

            time_values = res.time_values

            solution_data_info = res._solution_header(0)
            DOFS = solution_data_info['DOFS']
            
            ux = DOFS.index(1)
            uy = DOFS.index(2)
            uz = DOFS.index(3)
            
            if 'd' in out_quant:
                all_disp_d = res.nodal_time_history('NSL')[1][:, :, (ux, uy, uz)]
            else:
                all_disp_d = None
            if 'a' in out_quant:
                all_disp_a = res.nodal_time_history('ACC')[1][:, :, (ux, uy, uz)]
            else:
                all_disp_a = None
            if 'v' in out_quant:
                all_disp_v = res.nodal_time_history('VEL')[1][:, :, (ux, uy, uz)]
            else:
                all_disp_v = None

        if len(time_values) != timesteps:
            warnings.warn(f'The number of response values {len(time_values)} differs from the specified number of timesteps {timesteps} -> Convergence or substep errors.')

        t_end = t_start
        t_start = time.time()
        logger.info(f'RST parsing in {t_start-t_end} s')

        return time_values, [all_disp_d, all_disp_v, all_disp_a]

#     def mode_superpos(self, f=None, d=None):
#         ansys = self.ansys
#         # Transient/Harmonic Response
#         deltat=self.deltat
#         timesteps=self.timesteps
#
#         ansys.run('/SOL')
#         ansys.antype('TRANS')
#         ansys.trnopt(method='MSUP')# bug: vaout should be tintopt
#
#
#         ansys.kbc(0) # Specifies ramped or stepped loading within a load step.
#         ansys.timint(1) #Turns on transient effects.
#         ansys.alphad(value=0)
#         ansys.betad(value=0)
#         ansys.dmprat(ratio=0)
#         ansys.mdamp(stloc=1, v1=0, v2=0, v3=0, v4=0, v5=0, v6=0)
# #         ansys.outres(item='ERASE')
#         ansys.outres(item='ALL',freq='ALL')# Enable all output
# #         ansys.nsel(type='S', item='NODE', vmin=30,vmax=40) # select only nodes of interest
# #         #ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
# #         #ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
# #         ansys.nsel(type='ALL')
# #         ansys.outres(#item='A',
# #                      item='NSOL',
# #                      freq='ALL'
# #                      #,cname='meas_nodes'
# #                      )# Controls the solution data written to the database.
# #         ansys.outres(item='A',
# #                      #item='NSOL',
# #                      freq='ALL'
# #                      #,cname='meas_nodes'
# #                      )# Controls the solution data written to the database.
# #         ansys.outres(item='V',
# #                      #item='NSOL',
# #                      freq='ALL'
# #                      #,cname='meas_nodes'
# #                      )# Controls the solution data written to the database.
# #         ansys.rescontrol(action='DEFINE',ldstep='NONE',frequency='NONE',maxfiles=-1)  # Controls file writing for multiframe restarts
#         ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat, carry='OFF')
#
#         #ansys.solve()
#
#         t_end = deltat*(timesteps-1)
#         t = np.linspace(deltat,stop=t_end,num=timesteps)
#
#         printsteps = list(np.linspace(0,timesteps, 100, dtype=int))
#         dts=[]
#         timesteps=10000
#         t_start = time.time()
#         for lsnum in range(timesteps):
#             if not lsnum %1000:
#                 t_end=t_start
#                 t_start = time.time()
#                 dts.append(t_start-t_end)
#                 print(lsnum, np.mean(dts))
#
#             while lsnum in printsteps:
#                 del printsteps[0]
#                 print('.',end='', flush=True)
#             ansys.time((lsnum+1)*deltat)
#             if f is not None:
#                 ansys.fdele(node='ALL', lab='ALL')
#                 ansys.f(node=20,lab='FZ', value=f[lsnum])
#             if d is not None:
#                 ansys.ddele(node=20, lab='UZ')
#                 if d[lsnum]:
#                     ansys.d(node=20, lab='UZ', value=d[lsnum])
#             #continue
#             ansys.solve()
#         #np.savetxt('dts_amsupsolve',dts)
#         #asd
#         print('.',end='\n', flush=True)
#         ansys.finish()
#         ansys.run('/SOL')
#         ansys.expass(key='ON')
#         ansys.numexp(num=timesteps, begrng=0, endrng=timesteps*deltat)
#         ansys.solve()
#         ansys.finish()
#         self.last_analysis = 'trans'
#
#
#     def sweep(self, deltat = 0.01, timesteps = 1024, f_start=0/2/np.pi, f_end=None, phi=0, ampl= 1000):
#
#         nyq = 1/deltat/2.5
#         if f_end is None:
#             f_end=nyq
#
#         t_end = deltat*timesteps
#         t = np.linspace(deltat,stop=t_end,num=timesteps)
#
#         assert f_end<=nyq
#
#         f = np.sin(2*np.pi*((f_start*t)+(f_end-f_start)/(2*t_end)*t**2+phi))*ampl
#
#         self.deltat = deltat
#         self.timesteps = timesteps
#
#         return f

    def export_ans_mats(self):
        ansys = self.ansys
        jid = self.jobname
        # np.set_printoptions(precision=3, linewidth=200, suppress=True)
        ansys.slashsolu()
        ansys.antype('MODAL')
        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')
    
        ansys.nsel(type='ALL')
        ansys.outres(item='NSOL', freq='ALL')
        ansys.modopt(method='QRDAMP', nmode=100, freqb=0,
                     freqe=1e4,
                     cpxmod='cplx',
                     nrmkey='on',
                     )
        ansys.mxpand(nmode='all', elcalc=1)
        ansys.solve()
        ansys.wrfull()
        ansys.finish()
        ansys.aux2()
        # ansys.file(jid,'FULL')
        ansys.dmat(matrix="MatKD", type="D", method="IMPORT", val1="FULL", val2=f"{jid}.full", val3="STIFF")
        ansys.export(matrix="MatKD", format="MMF", fname="MatKD.bin")
        k = np.array(scipy.io.mmread('MatKD.bin'))
        ansys.dmat(matrix="MatMD", type="D", method="IMPORT", val1="FULL", val2=f"{jid}.full", val3="MASS")
        ansys.export(matrix="MatMD", format="MMF", fname="MatMD.bin")
        m = scipy.io.mmread('MatMD.bin')
        try:
            ansys.dmat(matrix="MatCD", type="D", method="IMPORT", val1="FULL", val2=f"{jid}.full", val3="DAMP")
            ansys.export(matrix="MatCD", format="MMF", fname="MatCD.bin")
            c = scipy.io.mmread('MatCD.bin')
        except Exception as e:
            # print(e)
            c = np.zeros_like(k)
        ansys.finish()
        
        # full_path = os.path.join(ansys.directory, ansys.jobname + '.full')
        # full = pyansys.read_binary(full_path)
        # # TODO: Check, that Nodes and DOFS are in the same order in modeshapes and k,m
        # dof_ref, k_, m_ = full.load_km(as_sparse=False, sort=False)
        # k_ += np.triu(k_, 1).T
        # m_ += np.triu(m_, 1).T
    # #     print(dof_ref)
        # for mode in range(num_modes):
            # msh_f = mshs[1:, 2, mode].flatten()
            #
            # kappa = msh_f.T.dot(k).dot(msh_f)
            # mu = msh_f.T.dot(m).dot(msh_f)
            #
            # print(np.sqrt(kappa / mu) / 2 / np.pi)
            #
            # msh_f = mshs[:, :, mode].flatten()
            #
            # kappa = msh_f.T.dot(k_).dot(msh_f)
            # mu = msh_f.T.dot(m_).dot(msh_f)
            # print(np.sqrt(kappa / mu) / 2 / np.pi)
        
        return k, m, c

    def save(self, save_dir):
        '''
        save under save_dir/{jobname}_mechanical.npz
        
        save enables to:
         - restore the ansys model
         - rerun analysis
         - retrieve previously run analyses
        '''
        logger.info(f'Saving Mechanical object to {save_dir}/{self.jobname}_mechanical.npz')
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        out_dict = {}
        # 0:build_mdof, 1:free_decay, 2:ambient, 3:impulse_response, 4:modal  
        
        out_dict['self.state'] = self.state
        
        out_dict['self.jobname'] = self.jobname
        
        if self.state[0]:
            out_dict['self.nodes_coordinates'] = self.nodes_coordinates
            #out_dict['self.num_nodes'] = self.num_nodes
            out_dict['self.num_modes'] = self.num_modes
            out_dict['self.d_vals'] = self.d_vals
            out_dict['self.k_vals'] = self.k_vals
            out_dict['self.masses'] = self.masses
            out_dict['self.eps_duff_vals'] = self.eps_duff_vals
            out_dict['self.sl_force_vals'] = self.sl_force_vals
            out_dict['self.hyst_vals'] = self.hyst_vals
            out_dict['self.meas_nodes'] = self.meas_nodes
            out_dict['self.damping'] = self.damping
        
        if self.state[1]:
            out_dict['self.decay_mode'] = self.decay_mode
            out_dict['self.t_vals_decay'] = self.t_vals_decay
            out_dict['self.resp_hist_decayd'] = self.resp_hist_decay[0]
            out_dict['self.resp_hist_decayv'] = self.resp_hist_decay[1]
            out_dict['self.resp_hist_decaya'] = self.resp_hist_decay[2]

        if self.state[2]:
            out_dict['self.inp_hist_amb'] = self.inp_hist_amb
            out_dict['self.t_vals_amb'] = self.t_vals_amb
            out_dict['self.resp_hist_ambd'] = self.resp_hist_amb[0]
            out_dict['self.resp_hist_ambv'] = self.resp_hist_amb[1]
            out_dict['self.resp_hist_amba'] = self.resp_hist_amb[2]

        if self.state[3]:
            out_dict['self.inp_hist_imp'] = self.inp_hist_imp
            out_dict['self.t_vals_imp'] = self.t_vals_imp
            out_dict['self.resp_hist_impd'] = self.resp_hist_imp[0]
            out_dict['self.resp_hist_impv'] = self.resp_hist_imp[1]
            out_dict['self.resp_hist_impa'] = self.resp_hist_imp[2]
            out_dict['self.modal_imp_energies'] = self.modal_imp_energies
            out_dict['self.modal_imp_amplitudes'] = self.modal_imp_amplitudes

        if self.state[4]:
            out_dict['self.damped_frequencies'] = self.damped_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.damped_mode_shapes'] = self.damped_mode_shapes
            out_dict['self.frequencies'] = self.frequencies
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.num_modes'] = self.num_modes
        
        if self.state[5]:
            out_dict['self.frequencies_comp'] = self.frequencies_comp
            out_dict['self.modal_damping_comp'] = self.modal_damping_comp
            out_dict['self.mode_shapes_comp'] = self.mode_shapes_comp
            
        if self.state[2] or self.state[3] or self.state[4] or self.state[6]:
            out_dict['self.trans_params'] = self.trans_params

            out_dict['self.deltat'] = self.deltat
            out_dict['self.timesteps'] = self.timesteps
        
        if self.state[6]:
            out_dict['self.t_vals_imp'] = self.t_vals_imp
            out_dict['self.IRF_matrix'] = self.IRF_matrix
            out_dict['self.imp_hist_imp_matrix'] = self.imp_hist_imp_matrix
            out_dict['self.modal_imp_energy_matrix'] = self.modal_imp_energy_matrix
            out_dict['self.modal_imp_amplitude_matrix'] = self.modal_imp_amplitude_matrix
        
        np.savez_compressed(os.path.join(save_dir, f'{self.jobname}_mechanical.npz'), **out_dict)
        
        
    @classmethod
    def load(cls, jobname, load_dir, ansys=None, wdir=None):
        
        assert os.path.isdir(load_dir)
        
        fname = os.path.join(load_dir, f'{jobname}_mechanical.npz')
        assert os.path.exists(fname)
        
        logger.info('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        
        assert jobname == in_dict['self.jobname'].item()
        
        mech = cls(ansys, jobname, wdir)
        
        state = list(in_dict['self.state'])
        
        if state[0]:
            nodes_coordinates = in_dict['self.nodes_coordinates']
            #print(nodes_coordinates, 'should be list of lists')
            #num_nodes = in_dict['self.num_nodes']
            num_modes = in_dict['self.num_modes'].item()
            d_vals = list(in_dict['self.d_vals'])
            k_vals = list(in_dict['self.k_vals'])
            masses = list(in_dict['self.masses'])
            eps_duff_vals = list(in_dict['self.eps_duff_vals'])
            sl_force_vals = list(in_dict['self.sl_force_vals'])
            hyst_vals = list(in_dict['self.hyst_vals'])
            meas_nodes = list(in_dict['self.meas_nodes'])
            #print(in_dict['self.damping'], type(in_dict['self.damping']))
            damping = in_dict['self.damping']
            if damping.size == 1:
                damping = damping.item()
            else:
                damping = list(damping)
                if len(damping) >= 2:
                    if damping[-1] == 1 or damping[-1] == 0:
                        damping[-1] = bool(damping[-1])
            #print(damping)
            #print(meas_nodes)
            mech.build_mdof(nodes_coordinates=nodes_coordinates,
                            k_vals=k_vals, masses=masses, d_vals=d_vals,
                            damping=damping, sl_force_vals=sl_force_vals,
                            eps_duff_vals=eps_duff_vals, hyst_vals=hyst_vals,
                            num_modes=num_modes, meas_nodes=meas_nodes)
        
        if state[1]:
            mech.decay_mode = in_dict['self.decay_mode'].item()
            mech.t_vals_decay = in_dict['self.t_vals_decay']
            mech.resp_hist_decay = in_dict['self.resp_hist_decay']

        if state[2]:
            mech.inp_hist_amb = in_dict['self.inp_hist_amb']
            mech.t_vals_amb = in_dict['self.t_vals_amb']
            mech.resp_hist_amb = in_dict['self.resp_hist_amb']

        if state[3]:
            mech.inp_hist_imp = in_dict['self.inp_hist_imp']
            mech.t_vals_imp = in_dict['self.t_vals_imp']
            mech.resp_hist_imp = in_dict['self.resp_hist_imp']
            mech.modal_imp_energies = in_dict['self.modal_imp_energies']
            mech.modal_imp_amplitudes = in_dict['self.modal_imp_amplitudes']

        if state[4]:
            mech.damped_frequencies = in_dict['self.damped_frequencies']
            mech.modal_damping = in_dict['self.modal_damping']
            mech.damped_mode_shapes = in_dict['self.damped_mode_shapes']
            mech.frequencies = in_dict['self.frequencies']
            mech.mode_shapes = in_dict['self.mode_shapes']
            mech.num_modes = in_dict['self.num_modes']
        
        if state[5]:
            mech.frequencies_comp = in_dict['self.frequencies_comp']
            mech.modal_damping_comp = in_dict['self.modal_damping_comp']
            mech.mode_shapes_comp = in_dict['self.mode_shapes_comp']
            
        if state[2] or state[3] or state[4] or state[6]:
            trans_params = in_dict['self.trans_params']
            if trans_params.size > 1:
                mech.trans_params = tuple(trans_params)
                
            mech.deltat = in_dict['self.deltat'].item()
            mech.timesteps = in_dict['self.timesteps'].item()
        
        if state[6]:
            mech.t_vals_imp = in_dict['self.t_vals_imp']
            mech.IRF_matrix = in_dict['self.IRF_matrix']
            mech.imp_hist_imp_matrix = in_dict['self.imp_hist_imp_matrix']
            mech.modal_imp_energy_matrix = in_dict['self.modal_imp_energy_matrix']
            mech.modal_imp_amplitude_matrix = in_dict['self.modal_imp_amplitude_matrix']
        
        mech.state = state
        
        return mech
        
    def get_geometry(self):
        '''
        return (meas)nodes, lines, chan_dofs in a format usable in pyOMA
        '''
        nodes = []
        for meas_node in np.concatenate(([1], self.meas_nodes)):
            for node, x, y, z in nodes:
                if node == meas_node:
                    nodes.append([node, x, y, z])
                    break
            else:
                logger.warning(f'Meas node {meas_node} was not found in nodes_coordinates')
        
        lines = []
        meas_node_last = 1  # ansys starts at 1
        for meas_node in self.meas_nodes:
            # how is that node connected to any other node in self.meas_nodes
            # for all_occurences_of_it_in_ntn_conns:
            #     for connect_level in range(num_nodes):
            #         find all nodes connected to it in ntn_conns
            #         check, if any of them are in meas_nodes
            #            store and remove from
            #            pah, that sucks
            lines.append((meas_node_last, meas_node))
        
        chan_dofs = []
        for channel, meas_node in enumerate(self.meas_nodes):
            chan_dofs.append((channel, meas_node, 90, 0))
        
        return nodes, lines, chan_dofs
    
    def export_geometry(self, save_dir='/usr/scratch4/sima9999/work/modal_uq/datasets/'):
        'save under jid_folder, nodes_file, lines_file, chan_dofs_file'
        os.makedirs(save_dir, exist_ok=True)
    
        nodes, lines, chan_dofs = self.get_geometry()
        
        with open(os.path.join(save_dir, 'grid.txt'), 'wt') as f:
            f.write('node_name\tx\ty\tz\n')
            for node, x, y, z in nodes:
                f.write(f'{node}\t{x:e}\t{y:e}\t{z:e}\n')
                
        with open(os.path.join(save_dir, 'lines.txt'), 'wt') as f:
            f.write('node_name_1\tnode_name_2\n')
            for line_s, line_e in lines:
                f.write(f'{line_s}\t{line_e}\n')
                
        with open(os.path.join(save_dir, 'chan_dofs.txt'), 'wt') as f:
            f.write('Channel-Nr.\tNode\tAzimuth\tElevation\tChannel Name\n')
            for channel, meas_node, az, elev in chan_dofs:
                f.write(f'{channel}\t{meas_node}\t{az}\t{elev}\t \n')
                
        return

    def numerical_response_parameters(self, num_modes=None, compensate=True, dofs = [2]):
        '''optionally, compensate with time integration parameters
        compensate: frequencies for spatial and temporal discretization(and mass matrix formulation?)
        compensate: damping ratios for non-constant rayleigh and temporal discretization
        modeshapes are always reduced to the set of meas nodes
        '''

        ndof = len(dofs)
        if num_modes is None:
            num_modes = self.num_modes
#         frequencies, _, _ = self.modal(damped=False, num_modes=num_modes)
        if self.trans_params is None and compensate:
            logger.info('No transient parameters set. Compensation will be skipped.')
            compensate = False
        
        frequencies, damping, mode_shapes = self.modal(damped=True, num_modes=num_modes)
        deltat = self.deltat
        
        meas_nodes = self.meas_nodes
        nodes_coordinates = self.nodes_coordinates
        meas_indices = []
        for meas_node in self.meas_nodes:
            for i, (node, x, y, z) in enumerate(nodes_coordinates):
                if node == meas_node:
                    meas_indices.append(i)
                    break
            else:
                raise RuntimeError(f'meas_node {meas_node} could not be found in nodes_coordinates')
        
        mode_shapes_n = np.full((len(meas_nodes) * ndof, num_modes), np.nan, dtype=complex)
        for mode in range(num_modes):
            for i, dof in enumerate(dofs):
                mode_shapes_n[len(meas_nodes) * i:len(meas_nodes) * (i + 1), mode] = mode_shapes[meas_indices, dof, mode]
                
        if not compensate:
            frequencies_n = frequencies
            damping_n = damping
        else:
            # (delta,alpha, alphaf,alpham)
            # gamma,beta, alphaf, alpham
            gamma_, beta_, alpha_f_, alpha_m_ = self.trans_params

            if gamma_ is None: gamma_ = 0
            if beta_ is None: beta_ = 0
            if alpha_f_ is None: alpha_f_ = 0
            if alpha_m_ is None: alpha_m_ = 0

#             h,beta,gamma,zeta,omega,Omega,eta=sympy.symbols('h \\beta \gamma \zeta \omega \Omega \eta', real=True, positive=True)
#             alpha_f, alpha_m, rho_inf=sympy.symbols('$\\alpha_f$ $\\alpha_m$ $\\rho_{\infty}$', real=True)
#
#             A1=sympy.Matrix([[1,0, -beta],[0, 1, -gamma],[Omega**2*(1-alpha_f),2*zeta*Omega*(1-alpha_f), 1-alpha_m]])
#             A2=sympy.Matrix([[1,1,0.5-beta],[0,1,1-gamma],[-Omega**2*(alpha_f),-2*zeta*Omega*(alpha_f), -alpha_m]])
#             A=A1.solve(A2)
            frequencies_n = np.full_like(frequencies, np.nan)
            damping_n = np.full_like(damping, np.nan)
#             for i,(zeta_, freq) in enumerate(zip(damping, frequencies)):
#                 Omega_ = deltat*freq
#                 A_sub = A.subs(beta, beta_).subs(gamma, gamma_).subs(alpha_f, alpha_f_).subs(alpha_m,alpha_m_).subs(zeta, zeta_)
#                 lamda = np.linalg.eigvals(np.array(A_sub.subs(Omega, Omega_)).astype(np.float64))
#                 rho = np.abs(lamda)
#                 phi = np.abs(np.angle(lamda))
#                 j = np.argmax((phi < np.pi)*rho)
#
#                 frequencies_n[i] = np.sqrt(phi[j]**2+np.log(rho[j])**2)
#                 damping_n[i] = -np.log(rho[j])/frequencies_n[i]

            for mode in range(num_modes):

                zeta_ = damping[mode]
                freq_ = frequencies[mode]
                omega_ = freq_ * 2 * np.pi  # damped response frequency, undamped would be a little higher
                # Omega_ = omega_ * deltat

                # If here the damped omega is provided, later Omega_hat_ud is to be used
                # If here the undamped omega is provided, later Omega_hat_dd is to be used
                # Providing damped omega is more logical, as this is the response frequency of the system,
                # also this would be inline with providing the physical parameters for the amplification matrix
                A1 = np.array([[1, 0, -beta_ * deltat ** 2],
                             [0, 1, -gamma_ * deltat  ],
                             [omega_ ** 2 * (1 - alpha_f_), 2 * zeta_ * omega_ * (1 - alpha_f_), 1 - alpha_m_      ]])
                A2 = np.array([[1, 1 * deltat, (0.5 - beta_) * deltat ** 2],
                             [0, 1, (1 - gamma_) * deltat    ],
                             [-omega_ ** 2 * (alpha_f_), -2 * zeta_ * omega_ * (alpha_f_), -alpha_m_            ]])

#                 zeta_ = damping[mode]
#                 freq_ = frequencies[mode]
#                 Omega_ = deltat*freq_*2*np.pi
#                 A1=np.array([[1,0, -beta_],[0, 1, -gamma_],[Omega_**2*(1-alpha_f_),2*zeta_*Omega_*(1-alpha_f_), 1-alpha_m_]])
#                 A2=np.array([[1,1,0.5-beta_],[0,1,1-gamma_],[-Omega_**2*(alpha_f_),-2*zeta_*Omega_*(alpha_f_), -alpha_m_]])
                A = np.linalg.solve(A1, A2)
                lamda = np.linalg.eigvals(A)
                lamda = lamda[np.logical_and(np.real(lamda) != 0, np.imag(lamda) != 0)]
                # find the conjugate eigenvalues
                if np.logical_and(np.real(lamda) < 0, np.imag(lamda) == 0).all():
                    logger.warning("System has only negative, real eigenvalues!")
                    frequencies_n[mode] = np.nan
                    damping_n[mode] = np.nan
                else:
                    loglamda = np.log(lamda)
                    if (np.imag(loglamda) > 0).any():
                        this_loglamda = loglamda[loglamda.imag > 0].max()
                        Omega_hat_dd = np.imag(this_loglamda)  # contains damping twice, as the damped sampling frequency was provided to the amplification matrix
                        Omega_hat_ud = np.abs(this_loglamda)  # undamped sampling frequency for damped response frequency, contains numerical damping

                        zeta_hat = -np.real(this_loglamda) / Omega_hat_ud  # contains both physical and numerical damping

                        frequencies_n[mode] = Omega_hat_ud / 2 / np.pi / deltat
                        damping_n[mode] = zeta_hat

        self.frequencies_comp = frequencies_n
        self.modal_damping_comp = damping_n
        self.mode_shapes_comp = mode_shapes_n
        
        self.state[5] = True
        
        return frequencies_n, damping_n, mode_shapes_n

    # def get_signal(self, channel_defs=None):
        # '''
        # TODO :
        # - for re-usability: the initial signal generation should include all meas_nodes, DOF and quantities, where meas_nodes may be all nodes or at least a large enough subset of these
        # - create a channel definition list : channel: meas node, dof, quantity (np.ndarray, dtype=int)
        # - function get_signal returns the requested (or most recent) time series reduced by channel definition list
        # '''
        #
        #
        #
        # return t_vals, signal, channel_defs


#
#     def impulse(self, deltat = 0.01, timesteps = 1024, imp_len = 10, ampl=10000):
#
#         nyq = 1/deltat/2
#
#         t_end = deltat*timesteps
#         t = np.linspace(deltat,stop=t_end,num=timesteps)
#         # Impulse of length imp_len*deltat
#
#         f=np.sin(np.linspace(0,np.pi,imp_len, endpoint=True))*ampl
#         f = np.concatenate((f, np.zeros(timesteps-imp_len)))
#
#         self.deltat = deltat
#         self.timesteps = timesteps
#
#         return f
#
#     def white_noise(self, deltat = 0.01, timesteps = 1024, ampl=1000):
#
#         # white noise
#         f = np.random.randn(timesteps)*ampl
#
#         self.deltat = deltat
#         self.timesteps = timesteps
#
#         return f
# def scatterplot_matrix(data, names, **kwargs):
    # import itertools
    # """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    # against other rows, resulting in a nrows by nrows grid of subplots with the
    # diagonal subplots labeled with "names".  Additional keyword arguments are
    # passed on to matplotlib's "plot" command. Returns the matplotlib figure
    # object containg the subplot grid."""
    # data = np.array(data)
    # numvars, numdata = data.shape
    # fig, axes = plot.subplots(nrows=numvars, ncols=numvars, figsize=(12, 12))
    # fig.subplots_adjust(hspace=0.05, wspace=0.05)
    #
    # for ax in axes.flat:
        # # Hide all ticks and labels
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        #
        # # Set up ticks only on one side for the "edge" subplots...
        # if ax.is_first_col():
            # ax.yaxis.set_ticks_position('left')
        # if ax.is_last_col():
            # ax.yaxis.set_ticks_position('right')
        # if ax.is_first_row():
            # ax.xaxis.set_ticks_position('top')
        # if ax.is_last_row():
            # ax.xaxis.set_ticks_position('bottom')
            #
    # # Plot the data.
    # for i, j in zip(*np.triu_indices_from(axes, k=1)):
        # for x, y in [(i, j), (j, i)]:
            # axes[y, x].plot(data[x], data[y], ls='none', marker='.', **kwargs)
    # for ij in range(numvars):
        # # print(names[ij])
        # # print(data[ij])
        # axes[ij, ij].hist(data[ij], bins=20)
        #
    # # Label the diagonal subplots...
    # for i, label in enumerate(names):
        # axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                # ha='center', va='center')
                #
    # # Turn on the proper x or y axes ticks.
    # for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        # axes[j, i].xaxis.set_visible(True)
        # axes[i, j].yaxis.set_visible(True)
        #
    # return fig


def response_frequency(time_values, ydata, p0=[1, 0.05, 2 * np.pi, 0]):
#     res=pyansys.read_binary(path)
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
#     #print(nnum)
#
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values

    popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata=this_t, ydata=ydata, p0=p0)  # ,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
    perr = np.sqrt(np.diag(pcov))
    # print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
    return popt, perr

# def process(path, last_analysis = 'trans', f = None):
#
#
#     '''
#     nsol,2,2,U,Z,uz
#     store,merge
#     plvar,2
#
#     nsol,3,2,ACC,Z,acz
#     store,merge
#     plvar,3
#     '''
#
#     res=pyansys.read_binary(path)
#
#     if last_analysis=='static': #static
#         nodes, disp = res.nodal_solution(0)
#         uz=disp[nodes==20, 2]#knoten 20, DOF 2 (UZ)
#         print(uz)
#         return uz
#
#     elif last_analysis == 'modal': #modal
#         num_modes = res.nsets#_resultheader['nsets']
#         if res._resultheader['cpxrst']: # real and imaginary parts are saved as separate sets
#             num_modes //= 2
#         nnodes = res._resultheader['nnod']
#         ndof = res._resultheader['numdof']
#
#         mode_shapes = np.full((nnodes,ndof,num_modes), (1+1j)*np.nan, dtype=complex)
#         frequencies = np.full(num_modes, np.nan)
#         damping = np.full(num_modes, np.nan)
#
#         if res._resultheader['cpxrst']:
#             for mode in range(num_modes):
#                 sigma = res.time_values[2*mode]
#                 omega = res.time_values[2*mode+1]
#                 if omega < 0 : continue # complex conjugate pair
#
#                 frequencies[mode] = omega
#                 damping[mode] = -sigma/np.sqrt(sigma**2+omega**2)
#
#                 mode_shapes[:,:,mode].real= res.nodal_solution(2*mode)[1]
#                 mode_shapes[:,:,mode].imag= res.nodal_solution(2*mode+1)[1]
#
#         else:
#             frequencies[:] = res.time_values
#             for mode in range(num_modes):
#                 nnum, modal_disp = res.nodal_solution(mode)
#                 mode_shapes[:,:,mode]= modal_disp
#
#         return frequencies, damping, mode_shapes
#
#     elif last_analysis == 'trans': #transient
#
#         #meas_nodes=res.geometry['components']['MEAS_NODES']
#         node_numbers=res.geometry['nnum']
#         num_nodes=len(node_numbers)
#         time_values=res.time_values
#         #print(time_values)
#         dt = time_values[1]-time_values[0]
#         time_values -= dt
#         #print(dt)
#         #print(res._resultheader['neqv'])
#
#         solution_data_info = res.solution_data_info(0)
#         DOFS = solution_data_info['DOFS']
#
#         uz = DOFS.index(3)
#         nnum, all_disp = res.nodal_time_history('NSL')
#
#         #print(nnum, all_disp.shape)
#         if f is not None:
#             #fix,axes = plot.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
#             fig = plot.figure()
#             ax1 = fig.add_subplot(221)
#             ax2 = fig.add_subplot(222, sharey=ax1)
#             ax3 = fig.add_subplot(223, sharex=ax1)
#             ax4 = fig.add_subplot(224)
#             t = time_values#np.linspace(0,f.shape[0]*dt,f.shape[0])
#             ax2.plot(t,f, marker='+')
#         for node in range(1,num_nodes):
#             if f is not None:
#
#                 ydata = all_disp[:,node,uz]
#                 #ydata = np.concatenate(([1],ydata))
#                 this_t = time_values
#                 #this_t = np.concatenate(([0],time_values))
#
#                 ax1.plot(all_disp[:,node,uz],f, label=str(nnum[node]), marker='+')
#
#                 ax3.plot(ydata,this_t, marker='+')
#                 popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = this_t, ydata=ydata, p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
#                 perr = np.sqrt(np.diag(pcov))
#                 print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
#                 print(perr)
#                 print(popt)
#                 this_t = np.linspace(0,time_values[-1],len(time_values)*10)
#                 ax3.plot(free_decay(this_t,*popt),this_t)
#
#                 Sxx = np.abs(np.fft.rfft(all_disp[:,node,uz]))
#                 freq = np.fft.rfftfreq(all_disp.shape[0], dt)
#                 #freq, Sxx = scipy.signal.welch(all_disp[:,node,uz], fs=1/dt, nperseg = f.shape[0]//2)
#                 ax4.plot(freq, Sxx, marker='+')
#                 ax4.axvline(20.0)
#                 print(freq[Sxx.argmax()])
#             else:
#                 plot.plot(all_disp[:,node,uz], label=str(nnum[node]))
#
#         ax = ax1
#         #ax.grid(True)
#         ax.legend()
#         ax.spines['left'].set_position('zero')
#         ax.spines['right'].set_color('none')
#         ax.spines['bottom'].set_position('zero')
#         ax.spines['top'].set_color('none')
#         ax.set_xlabel('y [m]')
#         ax.xaxis.set_label_coords(1.05, 0.55)
#         ax.set_ylabel('F [N]', rotation='horizontal')
#         ax.yaxis.set_label_coords(0.45, 1.05)
#
#         xmin,xmax = ax.get_xlim()
#         xlim = max(-1*xmin,xmax)
#         ax.set_xlim((-1*xlim,xlim))
#
#         ymin,ymax = ax.get_ylim()
#         ylim = max(-1*ymin,ymax)
#         ax.set_ylim((-1*ylim,ylim))
#         plot.show()
#     return
#
#     path='/dev/shm/test.csv'
#     t,d=[],[]
#     f=open(path,'rt')
#     f.readline()
#     for line in f:
#         l=line.split()
#         t.append(float(l[0]))
#         d.append(float(l[1]))
#
#     d=np.array(d)
#     t=np.array(t)
#     blocks=1
#     block_length = int(np.floor(len(t)/blocks))
#     H=np.zeros(block_length,dtype=complex)
#     for i in range(blocks):
#         Sx=np.fft.fft(d[i*block_length:(i+1)*block_length])
#         F=np.sin(t[i*block_length:(i+1)*block_length]*2*np.pi*t[i*block_length:(i+1)*block_length]/1000)
#         Sf=np.fft.fft(F)
#         Sff=Sf**2
#         Sfx=Sf*Sx
#         H+=Sfx/Sff
#     H/=blocks
#     fftfreq =np.fft.fftfreq(block_length, t[1]-t[0])
#     fig,axes=plot.subplots(2,1,sharex='col')
#     axes[0].plot(fftfreq, np.abs(H))
#     axes[1].plot(fftfreq, np.angle(H)/np.pi*180)
#     plot.xlim(xmin=0)
#     plot.show()
#
#     plot.plot(t,d)
#     plot.plot(t,np.sin(t*2*np.pi*t/1000))
#
#     plot.show()


def free_decay(t, R, zeta, omega_d, phi=0):
    return R * np.exp(-zeta * omega_d / (np.sqrt(1 - zeta ** 2)) * t) * np.cos(omega_d * t + phi)


def free_decay_acc(t, R, zeta, omega_d, phi=0):

    # return -2*R*omega_d**2*np.exp(-zeta*omega_d/(np.sqrt(1-zeta**2))*t)*np.cos(omega_d*t+phi)
    return -R * omega_d ** 2 * np.exp(-zeta * omega_d / (np.sqrt(1 - zeta ** 2)) * t) * np.cos(omega_d * t + phi)

# def generate_student(ansys, omega, zeta, d0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None, f_scale=None, **kwargs):
#     print(jid)
#     assert deltat is not None or dt_fact is not None
#     assert timesteps is not None or num_cycles is not None
#
#     m    = 1    # kg
#     k_2  = omega**2*m/(1-zeta**2)# N/m
#     d=zeta*(2*np.sqrt(k_2*m))
#
#     f_max = np.sqrt((k_2)/m)/np.pi/2
#
#     if dt_fact is None:
#         dt_fact =deltat*f_max # \varOmega /2/pi
#         print("\\varOmega",dt_fact*2*np.pi)
#     elif deltat is None:
#         deltat = dt_fact/f_max
#
#     if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
#
#     if timesteps is None:
#         timesteps = int(np.ceil(num_cycles/f_max/deltat))
#     elif num_cycles is None:
#         num_cycles = int(np.floor(timesteps*f_max*deltat))
#     if d0 is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, d0, deltat, dt_fact, timesteps, num_cycles))
#     elif f_scale is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, random excitation variance {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, f_scale, deltat, dt_fact, timesteps, num_cycles))
#     mech = Mechanical(ansys)
#     voigt_kelvin=mech.voigt_kelvin(k=k_2, d=d)
#     mass = mech.mass(m)
#     mech.build_sdof(
#                mass=mass,
#                voigt_kelvin = voigt_kelvin,
#                d_init=d0
#                )
#
#     mech.modal(True)
#
#     f_max = process(f'{jid}.rst', last_analysis=mech.last_analysis)[0].max()
#     f_analytical = np.sqrt((k_2)/m)/np.pi/2
#     zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#     #print(f_analytical,zeta_analytical)
#
#     mech.timesteps=timesteps
#     mech.deltat = deltat
#
#     #print(timesteps,deltat)
#
#     f=np.zeros(timesteps)
#     if d0 is not None:
#         f[0]=d0
#         mech.transient(d=f,parameter_set= kwargs.pop('parameter_set',None))
#     else:
#         assert f_scale is not None
#         f = np.random.normal(loc=0.0,scale=f_scale, size=(timesteps,))
#         mech.transient(f=f,parameter_set= kwargs.pop('parameter_set',None))
#         #mech.mode_superpos(f)
#
#
#     res=pyansys.read_binary(f"{jid}.rst")
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
#     #print(all_disp)
#
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
# #     print(ydata)
#     this_t = time_values
#     ty=np.vstack((this_t, ydata))
#     plot.figure()
#
#     plot.gca().plot(this_t, ydata, color='black', **kwargs)
#     plot.plot(this_t, ydata, ls='none', marker='+')
#     plot.show()
#
#     np.savetxt("/vegas/users/staff/womo1998/data_hadidi/{}.csv".format(jid), ty.T)
#     with open("/vegas/users/staff/womo1998/data_hadidi/description_new.txt", "at") as f:
#         f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(jid,k_2,d,deltat))
#
# def generate_student_nl(ansys, omega, zeta, d0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None, f_scale=None, save=True, **kwargs):
#
#     global jid
#     oldjid= jid
#     jid=str(uuid.uuid4()).split('-')[-1]
#     ansys.filname(fname=jid, key=1)
#     for file in glob.glob(f'/dev/shm/womo1998/{oldjid}.*'):
#         print(f'removing {file}')
#         os.remove(file)
#     print(jid)
#
#     total, used, free = shutil.disk_usage("/dev/shm/")
#     if free/total < 0.1:
#         raise RuntimeError(f'Disk "/dev/shm/ almost full {used} of {total}')
#
#     assert deltat is not None or dt_fact is not None
#     assert timesteps is not None or num_cycles is not None
#
#     m    = 1    # kg
#     k_2  = omega**2*m/(1-zeta**2)# N/m
#     d=zeta*(2*np.sqrt(k_2*m))
#
#     f_max = np.sqrt((k_2)/m)/np.pi/2
#
#     if dt_fact is None:
#         dt_fact =deltat*f_max # \varOmega /2/pi
#         print("\\varOmega",dt_fact*2*np.pi)
#     elif deltat is None:
#         deltat = dt_fact/f_max
#
#     if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
#
#     nl_ity=kwargs.pop('nl_ity',0)
#
#     if timesteps is None:
#         timesteps = int(np.ceil(num_cycles/f_max/deltat))
#     elif num_cycles is None:
#         num_cycles = int(np.floor(timesteps*f_max*deltat))
#     if d0 is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}, nonlinearity {}".format(omega, zeta, d0, deltat, dt_fact, timesteps, num_cycles, nl_ity))
#     elif f_scale is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, random excitation variance {:1.3f}, deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}, nonlinearity {}".format(omega, zeta, f_scale, deltat, dt_fact, timesteps, num_cycles,nl_ity))
#
#     if d0 is None:
#         sigma_scale = 1 # 68–95–99.7 rule
#         empiric_factor=1/2.5
#         d_max = sigma_scale*f_scale/2/k_2/zeta*empiric_factor
#     else:
#         d_max = d0
#
#     mech = Mechanical(ansys)
#     voigt_kelvin=mech.voigt_kelvin(k=k_2*(1-nl_ity), d=d)
#     nonlinear = mech.nonlinear(nl_ity=nl_ity, d_max=d_max, k_lin=k_2)
#     mass = mech.mass(m)
#     mech.build_sdof(
#                mass=mass,
#                voigt_kelvin = voigt_kelvin,
#                nonlinear = nonlinear,
#                d_init=d0
#                )
#
#     mech.modal(True)
#
#     f_max = process(f'{jid}.rst', last_analysis=mech.last_analysis)[0].max()
#     f_analytical = np.sqrt((k_2)/m)/np.pi/2
#     zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#     #print(f_analytical,zeta_analytical)
#
#     mech.timesteps=timesteps
#     mech.deltat = deltat
#
#     #print(timesteps,deltat)
#
#     f=np.zeros(timesteps)
#     if d0 is not None:
#         f[0]=d0
#         mech.transient(d=f,parameter_set= kwargs.pop('parameter_set',None))
#     else:
#         assert f_scale is not None
#         f = np.random.normal(loc=0.0,scale=f_scale, size=(timesteps,))
#         mech.transient(f=f,parameter_set= kwargs.pop('parameter_set',None))
#         #mech.mode_superpos(f)
#
#
#     res=pyansys.read_binary(f"{jid}.rst")
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
# #     print(all_disp)
#
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
# #     print(ydata)
#     this_t = time_values
#     ty=np.vstack((this_t, ydata))
#     plot.figure()
#
#     plot.gca().plot(this_t, ydata, color='black', **kwargs)
#     plot.axhline(d_max)
#     plot.axhline(-d_max)
# #     plot.plot(this_t, ydata, ls='none', marker='+')
#     #plot.show()
#     #print("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(jid,omega,zeta,d0,deltat,nl_ity))
#     source_folder = "/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_"
#     if d0 is not None:
#         source_folder+='decay/'
#     else:
#         source_folder +='ambient/'
#
#     if save:
#         np.savetxt(f"{source_folder}{jid}.csv", ty.T)
#         np.savetxt(f"{source_folder}inp{jid}.csv",f)
#         with open(f"{source_folder}description.txt", "at") as f:
#             f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(jid,k_2,d,d_max,deltat,nl_ity))
#


def generate_sdof_time_hist(ansys,
                            omega, zeta, m=1, fric_visc_rat=0, nl_ity=0,  # structural parameters
                            dscale=None, f_scale=None,  # loading parameters
                            deltat=None, dt_fact=None, timesteps=None, num_cycles=None, num_meas_nodes=None, meas_nodes=None,  # signal parameters
                            savefolder=None, working_dir='/dev/shm/womo1998/', jid=None,  # function parameters
                            ** kwargs):
#                      ansys, jid, working_dir='/dev/shm/womo1998/', # function parameters
#                             omega, zeta, m=1, fric_visc_rat=0, nl_ity=0, # structural parameters
#                             d0=None, f_scale=None,  # loading parameters
#                             deltat=None, dt_fact=None, timesteps=None, num_cycles=None, # signal parameters
#                             **kwargs):
    assert np.abs(nl_ity) <= 0.5
    assert fric_visc_rat <= 1
    try:
        ansys.finish()
    except pyansys.errors.MapdlExitedError as e:
        logger.exception(e)
        ansys = Mechanical.start_ansys()
    ansys.clear()

    # compute global structural parameters
    k = omega ** 2 * m / (1 - zeta ** 2)  # N/m
    d = 2 * zeta * np.sqrt(k * m)

    # Compute maximum displacement from expected excitation parameters and structural parameters
    # this is used mainly for computation of the nonlinear springs, where dmax should not be exceeded
    if f_scale is not None:  # assuming gaussian white noise excitation
        sigma_scale = 1  # 68–95–99.7 rule
        empiric_factor = 1 / 2.5
        d_max = sigma_scale * f_scale / 2 / k / max(zeta, 1 / 1000) * empiric_factor
    elif dscale is not None:  # assuming free decay
        d_max = dscale
        # TODO: If free-decay whith white noise the maximum of both should be taken
    else:
        d_max = kwargs.pop('d_max', None)
        if d_max is None:
            raise RuntimeError('Neither d0 nor f_scale were defined.')

    # Compute expected absolute displacement from excitation parameters and structural parameters
    # this is used mainly for computation of equivalent friction damping, which should be equivalent on average
    # https://en.wikipedia.org/wiki/Average_absolute_deviation
    # ratio of mean absolute deviation to standard deviation is sqrt(2 / π) = 0.79788456...
    if f_scale is not None:  # assuming gaussian white noise excitation
        # dynamic amplification factor https://en.wikipedia.org/wiki/Structural_dynamics#Damping
        DAF = 1 + np.exp(-zeta * np.pi)
        d_mean = np.sqrt(2 / np.pi) * f_scale / k * DAF
    elif dscale is not None:  # assuming free decay
        d_mean = dscale
        # TODO: If free-decay whith white noise the maximum of both should be taken
    else:
        d_mean = kwargs.pop('d_mean', None)
        if d_mean is None:
            raise RuntimeError('Neither d0 nor f_scale were defined.')

    # compute elemental parameters
    k_lin = k * (1 - nl_ity)
    if nl_ity:
        logger.warning("Should k_lin really be reduced here? check definitions of nonlinear again")
    d_visc = d * (1 - fric_visc_rat)
    d_fric = d * fric_visc_rat

    # generate Elements
    mech = Mechanical(ansys, jobname=jid, wdir=working_dir)

#     if kwargs.pop('remove_mass',False):mass = mech.mass(m*1e-5)
#     else: mass = mech.mass(m)

#     if fric_visc_rat == 0: # only viscous damping
#         voigt_kelvin=mech.voigt_kelvin(k=k_lin, d=d_visc)
#         coulomb = None
#     else:
    if fric_visc_rat > 0 and fric_visc_rat <= 1:

        k_1 = k_lin * 100

        # Computation of equivalent friction damping
        fsl_equiv = d_mean / 2 * (k_1 - np.sqrt(k_1 * (-np.pi * omega * d_fric + k_1)))
        logger.info(f"equivalent slip force {fsl_equiv} for d {d_fric} at omega {omega} and d_mean {d_mean} with k_1 {k_1}")
    else:
        fsl_equiv = None
#         coulomb = mech.coulomb(k_2=k_lin, k_1=k_1, f_sl= f_sl_equiv, d=d_visc)

#     if nl_ity!=0:
#         assert abs(nl_ity)<=0.5
#         nonlinear = mech.nonlinear(nl_ity, d_max, k)
#     else:
#         nonlinear = None

    # build final modal
#     mech.build_sdof(
#        mass=mass,
#        voigt_kelvin = voigt_kelvin,
#        coulomb = coulomb,
#        nonlinear = nonlinear,
#        d_init=d0
#        )
    mech.build_mdof(nodes_coordinates=[(1, 0, 0, 0), (2, 0, 0, 0)],
                   k_vals=[k_lin], masses=[1, 1], d_vals=[d_visc], damping=None,
                   sl_forces=[fsl_equiv], nl_rats=[nl_ity], d_max=[0, d_max],
                   num_modes=1, meas_nodes=[2], **kwargs)

    # check modal results match with expected parameters
    # TODO: account for nonlinear stiffness
    try:
        freqs, damping, _ = mech.modal(True)
        logger.info(f"Numeric solution: f={freqs} zeta={damping*100}")
    except Exception as e:
        err_file = os.path.join(ansys.directory, ansys.jobname + '.err')
        with open(err_file) as f:
            lines_found = []
            block_counter = -1
            while len(lines_found) < 10:
                try:f.seek(block_counter * 4098, os.SEEK_END)
                except IOError:
                    f.seek(0)
                    lines_found = f.readlines()
                    break
                lines_found = f.readlines()
                block_counter -= 1
            logger.error(str(lines_found[-10:]))

            raise e
        logger.warning('Modal analysis failed, probably something is wrong with the model.')

    f_analytical = np.sqrt((k) / m) / np.pi / 2
    zeta_analytical = d / 2 / np.sqrt(k * m) * 100  # in percent of critical damping
    logger.info(f"Analytic solution: f={f_analytical},zeta={zeta_analytical}")

#     # Setup transient simulation parameters
#     assert deltat is not None or dt_fact is not None
#     assert timesteps is not None or num_cycles is not None
#
#     f_max = omega/np.pi/2
#
#     if dt_fact is None:
#         dt_fact =deltat*f_max
#     elif deltat is None:
#         deltat = dt_fact/f_max
#
#     if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")

    if dscale is not None:
        t_vals, resp_hist = mech.free_decay(0, dscale, dt_fact=dt_fact, num_cycles=num_cycles, **kwargs)
    elif f_scale is not None:
        t_vals, resp_hist, inp_hist = mech.ambient(f_scale, dt_fact=dt_fact, num_cycles=num_cycles, **kwargs)
    else:
        logger.info(f"Simulating system with omega {omega:1.3f}, zeta {zeta:1.3f}, fric_visc_rat {fric_visc_rat:1.3f}, nonlinearity {nl_ity}, user_provided excitation, deltat {deltat:1.5f}, dt_fact {dt_fact:1.2f}, timesteps {timesteps}, num_cycles {num_cycles}")
        # provide a custom forcing function for verification purposes
        f = kwargs.pop('f', None)
        t_vals, resp_hist = mech.transient(f=f, **kwargs)

    if kwargs.pop('data_hadidi', None):

       return np.hstack((t_vals[:, np.newaxis], resp_hist[0])), k, d_visc, d_max, fsl_equiv, deltat

    return

# def generate_conti(ansys, deltat=None,timesteps=None, num_cycles=None, **kwargs):
#
#     assert deltat is not None
#     assert timesteps is not None or num_cycles is not None
#     try: os.remove('file.rst')
#     except: pass
#
#
#     parameters = {
#                 'L'         : 40,
#
#                 'E'         : 210e9,
#                 'A'         : 4.97e-3,
#                 'rho'       : 7850,
#                 'Iy'        : 15.64e-6,
#                 'Iz'        : 15.64e-6,
#
#                 'ky_nl'     : 0,
#                 'kz_nl'     : 0,
#                 'x_knl'     : 15,
#
#                 'm_tmd'     : 0,
#                 'ky_tmd'    : 0,
#                 'kz_tmd'    : 0,
#                 'dy_tmd'    : 0,
#                 'dz_tmd'    : 0,
#                 'x_tmd '    : 40,
#
#             }
#     initial = {'d0y'       : 0.1,
#                'd0z'       : 0.1,
#                'x_d0'      : 40}
#     mech = Mechanical(ansys)
#
#
#     mech.build_conti(parameters, Ldiv = 10, initial=initial
#                )
#
#     mech.modal(True)
#
# #     ansys.open_gui()
#     frequencies, damping, mode_shapes = process('file.rst', last_analysis=mech.last_analysis)
#
#
#     if timesteps is None:
#         timesteps = int(np.ceil(num_cycles/min(frequencies)/deltat))
#
#     num_cycles_min = int(np.floor(timesteps*min(frequencies)*deltat))
#     num_cycles_max = int(np.floor(timesteps*max(frequencies)*deltat))
#
#     print("Generating {} cycles at {:1.3f} Hz ({:1.3f} \% damping) and {} cycles at {:1.3f} Hz ({:1.3f} \% damping)@ deltat {:1.5f} ".format(num_cycles_min, min(frequencies),damping[frequencies.argmin()]*100, num_cycles_max, max(frequencies), damping[frequencies.argmax()]*100,deltat))
#
#
#     mech.timesteps= timesteps
#     mech.deltat = deltat
#
#     f=np.zeros(timesteps)
#     f[0]=d0
#     mech.transient(d=f,parameter_set='AAM')
# #     mech.transient(d=f,meth='G-alpha',parameter_set=0.5)
#
#     res=pyansys.read_binary("file.rst")
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#
#
#     #print(DOFS)
#     nnum, all_disp = res.nodal_time_history('NSL')
# #     print(nnum, all_disp[:,:,2])
#     nnum, all_vel = res.nodal_time_history('VEL')
# #     print(nnum, all_vel[:,:,2])
#     nnum, all_acc = res.nodal_time_history('ACC')
# #     print(nnum, all_acc[:,:,2])
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#     uz = DOFS.index(3)
#
#     #ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
#     this_t = time_values
#     #ty=np.vstack((this_t, ydata))
#
#     deltaf = 0.1
#     axes = kwargs.pop('axes')#, sharey='col')
#     if axes is None:
#         axes = plot.subplots(nrows = 3, ncols=2, sharex='col')[1]
# #     for node in range(2,3):
# #         axes[0,0].plot(this_t, all_disp[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
# #         axes[0,1].psd(all_disp[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
# #         axes[1,0].plot(this_t, all_vel[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
# #         axes[1,1].psd(all_vel[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
# #         axes[2,0].plot(this_t, all_acc[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
# #         axes[2,1].psd(all_acc[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
#     node = 2
#     axes[0].plot(this_t, all_acc[:,node,2], ls='solid')
#     axes[1].psd(all_acc[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf), label='$f_s = {:1.0f}$ \\si{{\\hertz}}'.format(1/deltat))
#
#     for freq in frequencies:
# #         axes[0,1].axvline(freq, color='grey')
# #         axes[1,1].axvline(freq, color='grey')
# #         axes[2,1].axvline(freq, color='grey')
#         axes[1].axvline(freq, color='grey')
#
#
#     axes[0].set_xlim((0, 1/min(frequencies)*4))
#
# #     axes[0,0].set_ylim((-1.1,1.1))
# #     axes[0,1].grid(False)
# #     axes[1,1].grid(False)
# #     axes[2,1].grid(False)
#
#     #plot.plot(this_t, ydata, ls='none', marker='+')
# #     plot.show()
#
# #     import uuid
# #     id=str(uuid.uuid4()).split('-')[-1]
# #     np.savez("/vegas/scratch/womo1998/vib_data/{}.npz".format(id), this_t = this_t, all_disp = all_disp, all_vel=all_vel, all_acc=all_acc)

# def accuracy_study(ansys):
#
#     timesteps=1000
#     results = np.zeros((40,8))
#
#     if False:
#         for i in range(40):
#             dt_fact = (i+1)*0.005
#
#             mech = Mechanical(ansys)
#
#             fact=1e5
#             m    = fact/1    # kg
#             k_2  = 1*fact*4*np.pi**2 # 3947841.76 N/m
#             ampl = fact/10   # N
#             d=0.00*2*np.sqrt(k_2*m) # 0
#
#             voigt_kelvin=mech.voigt_kelvin(k=k_2, d=d)
#             mass = mech.mass(m)
#             mech.build_sdof(
#                        mass=mass,
#                        voigt_kelvin = voigt_kelvin,
#                        d_init=1
#                        )
#
#             mech.modal()
#
#             f_max = process('file.rst', last_analysis=mech.last_analysis).max()
#             f_analytical = np.sqrt((k_2)/m)/np.pi/2
#             zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#             print(f_analytical,zeta_analytical)
#
#             deltat = dt_fact/f_max
#             dt_fact =deltat/f_max
#             num_cycles = 16
#             timesteps = int(np.ceil(num_cycles/dt_fact))
#
#             mech.timesteps=timesteps
#             mech.deltat = deltat
#
#             f=np.zeros(timesteps)
#             f[0]=1
#             mech.transient(d=f)
#
#             popt,perr = response_frequency('file.rst')
#             results[i,:4]=popt
#             results[i,4:]=perr
#
#
#         ansys.exit()
#         np.save('newmark_errors.npz', results)
#
#     results = np.load('newmark_errors.npz.npy')
#     dt_fact = np.linspace(0.005,0.2,40)
#     #print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
#
#     #confidence intervals
#     sqrt_ts = np.sqrt(np.ceil(16/dt_fact))
#     tppf = [scipy.stats.t.ppf(0.95,int(timesteps)) for timsteps in np.ceil(16/dt_fact)]
#     #scaling
#     results[results[:,0]<0,0]*=-1
#     results[:,0]+=-0.5
#     results[:,0]/=0.5
#
#     results[:,4]/=0.5**2
#     results[:,4]/=sqrt_ts
#     results[:,4]*=tppf
#
#
#     results[:,1]*=100
#
#     results[:,5]*=100**2
#     results[:,5]/=sqrt_ts
#     results[:,5]*=tppf
#
#
#     results[:,2]/=2*np.pi
#     results[:,2]-=1# to make it delta_frequency
#
#     results[:,6]/=(2*np.pi)**2
#     results[:,6]/=sqrt_ts
#     results[:,6]*=tppf
#
#
#     results[:,3]=np.arcsin(np.sin(results[:,3]))
#     results[:,3]*=180/np.pi
#     results[results[:,3]<=0,3]*=-1
#
#     results[:,7]*=(180/np.pi)**2
#     results[:,7]/=sqrt_ts
#     results[:,7]*=tppf
#
#     with matplotlib.rc_context(rc=print_context_dict):
#         fig,axes = plot.subplots(2,2, sharex=True)
#         print(fig.get_size_inches())
#         axes = axes.flatten()
#         ylabels = ['$\Delta R$ [\si{\percent}]', '$\Delta \zeta [\si{\percent}]$', '$\Delta f [\si{\percent}]$', '$\Delta \phi [\si{\degree}]$']
#
#         for j in range(4):
#
#             axes[j].errorbar(dt_fact, results[:,j], yerr=results[:,j+4], errorevery=2)
#             axes[j].set_ylabel(ylabels[j])
#             #axes[j].plot(dt_fact[:], results[:,j], marker='+')
#         axes[2].set_xlabel('Frequency ratio $\\nicefrac{f_{\\text{max}}}{f_s} [-]$')
#         axes[3].set_xlabel('Frequency ratio $\\nicefrac{f_{\\text{max}}}{f_s} [-]$')
#         plot.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.085, hspace=0.1, wspace=0.18)
#         plot.show(block=True)


def verify_friction(ansys):
    ########
    # Verification code for friction / generate_damping
    ########
    plot.subplot()
    zeta = 10 / 100
    num_cycles = 8
    d0 = 1
    omega = 2
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta, fric_visc_rat=0, d0=d0, deltat=0.03, num_cycles=8)
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta, fric_visc_rat=0.5, d0=d0, deltat=0.03, num_cycles=8)
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta, fric_visc_rat=1, d0=d0, deltat=0.03, num_cycles=8)
    plot.show()


def hysteresis(ansys):
    '''
    Generate a hysteresis curve for different types of damping and stiffness curves
    For generating a hysteresis, the spring/damper/slider system is displaced by a known force
    (typically a full sine cycle, where a first quarter cycle is needed to reach stead-state and is discarded
    any other forces should not be present, that means, the inertia forces of the mass should be removed
    two options exist: turning off transient effects (stepped static analysis) or removing the mass element
    when transient effects are turned off, velocities are not computed and therefore viscous damping forces are zero
    removing the mass causes the response to grow out of bounds, therefore mass is scaled by a factor
    '''

    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # free-decay time histories for 2 % damping over 10 cycles in one plot
    # linear-viscous, duffing degressive (viscous), duffing progressive (viscous), friction
    if False:
        if False:  # generate data
            omega = 1
            ydata1, ydata2, ydata3, ydata4 = [], [], [], []
            for zeta in [2, 5, 10]:
                ty1, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=0, d0=1, dt_fact=1 / 1000, num_cycles=10)
                ty2, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=-0.5, d0=1, dt_fact=1 / 1000, num_cycles=10)  # degressive
                ty3, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=0.5, d0=1, dt_fact=1 / 1000, num_cycles=10)  # progressive
                ty4, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=1, nl_ity=0, d0=1, dt_fact=1 / 1000, num_cycles=10)
                ydata1.append(ty1)
                ydata2.append(ty2)
                ydata3.append(ty3)
                ydata4.append(ty4)
            np.savez('time_hist_decay_different_systems.npz', ydata1=ydata1, ydata2=ydata2, ydata3=ydata3, ydata4=ydata4)
        else:
            arr = np.load('time_hist_decay_different_systems.npz')  #
            ydata1 = arr['ydata1']
            ydata2 = arr['ydata2']  # degressive
            ydata3 = arr['ydata3']  # progressive
            ydata4 = arr['ydata4']

        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plot.subplots(2, 2, True, True)
            for ty1, ty2, ty3, ty4, zeta, color in reversed(list(zip(ydata1, ydata2, ydata3, ydata4, [2, 5, 10], ['black', 'darkgrey', 'lightgray']))):

                axes.flat[0].plot(ty1[0, :], ty1[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)
                axes.flat[1].plot(ty2[0, :], ty2[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)
                axes.flat[2].plot(ty3[0, :], ty3[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)
                axes.flat[3].plot(ty4[0, :], ty4[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)

            axes.flat[0].set_title('Linear, viscous', fontdict={'fontsize':10})
            axes.flat[1].set_title('Duffing (degressive), viscous', fontdict={'fontsize':10})
            axes.flat[2].set_title('Duffing (progressive), viscous', fontdict={'fontsize':10})
            axes.flat[3].set_title('Linear, friction', fontdict={'fontsize':10})

            axes[0, 0].set_ylabel('y [\si{\metre}]')
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_ylabel('y [\si{\metre}]')
            axes[1, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_xlabel('t [\si{\second}]')
            # axes[1,0].xaxis.set_label_coords(0.49, -0.07)
            axes[1, 1].set_xlabel('t [\si{\second}]')
            # axes[1,1].xaxis.set_label_coords(0.49, -0.06)

            for ax in axes.flat:
                ax.xaxis.set_major_formatter(plot.FuncFormatter(
                    lambda val, pos: '${:d}\pi$'.format(int(val / np.pi)) if val != 0 else '0'))
                ax.xaxis.set_minor_locator(plot.MultipleLocator(2 * np.pi))
                ax.xaxis.set_major_locator(plot.MultipleLocator(4 * np.pi))
                ax.grid(True)
                ax.set_ylim((-1, 1))
                ax.set_xlim((0, 2 * np.pi * 10))
                ax.set_yticks([-1, -0.5, 0, 0.5, 1],)
                ax.set_yticklabels(['$-1$', '$-\\nicefrac{1}{2}$', '$0$', '$\\nicefrac{1}{2}$', '$1$'])

            fig.subplots_adjust(left=0.08, bottom=0.125, right=0.970, top=0.940, hspace=0.2, wspace=0.1)
            plot.savefig('/ismhome/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/time_hist_decay_different_systems.pdf')
            plot.savefig('/ismhome/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/time_hist_decay_different_systems.png')
            plot.show()

    # massless system loaded with 1.25 sine cycles with 1 %, 5% and 20 % damping (equivalent)
    # 4 plots of hysteresis for linear-viscous, duffing degressive (viscous), duffing progressive (viscous), friction

    if False:

        if False:
            omega = 1
            f = np.sin(np.linspace(0, 2.5 * np.pi, 1250)) * 1
            ydata1, ydata2, ydata3, ydata4 = [], [], [], []
            all_ydata = [ydata1, ydata2, ydata3, ydata4]

            for ydata, nl_ity in zip(all_ydata, [0, 0.5, -0.5]):  # linear,#progressive,#degressive

                for zeta in [2, 5, 10]:
                    # for viscous timint=1 is needed, therefore reduce mass by factor 1e-6 -> remove_mass=True
                    ty, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=nl_ity, d_max=1, dt_fact=1 / 1000, num_cycles=1.25, remove_mass=True, timint=1, f=f)
                    ydata.append(ty[1, :])

            ydata = ydata4
            for zeta in [1, 5, 20]:
                # for friction timint=0 and normal mass
                ty, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=1, nl_ity=0, d_max=1, dt_fact=1 / 1000, num_cycles=1.25, remove_mass=False, timint=0, f=f)
                ydata.append(ty[1, :])
            np.savez('hysteresis_different_systems.npz', f=f, ydata1=ydata1, ydata2=ydata2, ydata3=ydata3, ydata4=ydata4)
        else:
            arr = np.load('hysteresis_different_systems.npz')  #
            f = arr['f']
            ydata1 = arr['ydata1']
            ydataprog = arr['ydata2']  # progressive
            ydatadeg = arr['ydata3']  # degressive
            ydata4 = arr['ydata4']
        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            for ydata, ax in zip([ydata1, ydatadeg, ydataprog, ydata4], axes.flat):
                for y, zeta, color in reversed(list(zip(ydata, [2, 5, 10], ['black', 'darkgrey', 'lightgray']))):
                    x = f[250:]
                    y = y[250:]
                    area = PolyArea(x, y)
                    ax.plot(y, x, color=color, label=f'{zeta}', linewidth=1)
                # ax.legend()
            axes.flat[0].set_title('Linear, viscous', fontdict={'fontsize':10})
            axes.flat[1].set_title('Duffing (degressive), viscous', fontdict={'fontsize':10})
            axes.flat[2].set_title('Duffing (progressive), viscous', fontdict={'fontsize':10})
            axes.flat[3].set_title('Linear, friction', fontdict={'fontsize':10})

            axes[0, 0].set_ylabel('f [\si{\\newton}]')
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_ylabel('f [\si{\\newton}]')
            axes[1, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_xlabel('y [\si{\metre}]')
            # axes[1,0].xaxis.set_label_coords(0.49, -0.07)
            axes[1, 1].set_xlabel('y [\si{\metre}]')
            # axes[1,1].xaxis.set_label_coords(0.49, -0.06)

            for ax in axes.flat:
                # ax.grid(True)
                ax.set_yticks([-1, -0.5, 0, 0.5, 1],)
                ax.set_yticklabels(['$-1$', '$-\\nicefrac{1}{2}$', '$0$', '$\\nicefrac{1}{2}$', '$1$'])
                ax.set_xticks([-1, -0.5, 0, 0.5, 1],)
                ax.set_xticklabels(['$-1$', '$-\\nicefrac{1}{2}$', '$0$', '$\\nicefrac{1}{2}$', '$1$'])
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
            fig.subplots_adjust(left=0.08, bottom=0.125, right=0.970, top=0.940, hspace=0.2, wspace=0.1)
            plot.savefig('/ismhome/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/hysteresis_different_systems.pdf')
            plot.savefig('/ismhome/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/hysteresis_different_systems.png')
            plot.show()

    # effect of forcing amplitude and rate on the hysteresis with 5 % damping (equivalent)
    # 5 plots in a 3x3 subplot, where the middle is the base extending left/right/up/down
    # all four system in each subplot

    if False:

        if False:
            zeta = 20
            ydata1, ydata2, ydata3, ydata4 = [], [], [], []
            fdata = []
            # omega, f, axes
            for omega, f_factor in [(1, 2,),
                                    (0.5, 1),
                                    (1 , 1),
                                    (2, 1),
                                    (1, 0.5)]:

                # TODO: d_max=1 is actually wrong here for increased amplitudes
                f = np.sin(np.linspace(0, 2.5 * np.pi, 1250)) * f_factor
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=0, nl_ity=0, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=True, timint=1, f=f)
                ydata1.append(ty)
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=0, nl_ity=0.5, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=True, timint=1, f=f)
                ydata2.append(ty)
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=0, nl_ity=-0.5, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=True, timint=1, f=f)
                ydata3.append(ty)
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=1, nl_ity=0, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=False, timint=0, f=f)
                ydata4.append(ty)

                fdata.append(f)
            np.savez('hysteresis_different_forcing.npz', fdata=fdata, ydata1=ydata1, ydata2=ydata2, ydata3=ydata3, ydata4=ydata4)
        else:
            arr = np.load('hysteresis_different_forcing.npz')  #
            fdata = arr['fdata']
            ydata1 = arr['ydata1']
            ydataprog = arr['ydata2']  # progressive
            ydatadeg = arr['ydata3']  # degressive
            ydata4 = arr['ydata4']

        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plot.subplots(nrows=3, ncols=3, sharex=False, sharey=False)
            for ty1, ty2, ty3, ty4, f, ax in zip(ydata1, ydatadeg, ydataprog, ydata4, fdata, (axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 1])):
                x = f[250:]

                y = ty1[1, 250:]
                ax.plot(y, x, color='black', label=f'')
                y = ty2[1, 250:]
                # ax.plot(y,x, color='black', label=f'')
                y = ty3[1, 250:]
                # ax.plot(y,x, color='black', label=f'')
                y = ty4[1, 250:]
                # ax.plot(y,x, color='black', label=f'')

            for ax in axes.flat:
                ax.set_xlim((-2, 2))
                ax.set_ylim((-2, 2))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_color('none')
                # ax.spines['top'].set_color('none')

                if ax == axes[1, 1]:
                    ax.spines['left'].set_position('zero')
                    ax.spines['bottom'].set_position('zero')

                    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
                    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
                    # ax.axis['xzero'].set_axisline_style("-|>")
                    # ax.axis['yzero'].set_axisline_style("-|>")
                    ax.xaxis.set_label_coords(0.9, 0.45)
                    ax.yaxis.set_label_coords(0.45, 0.95)
                    ax.set_ylabel('f [\si{\\newton}]')
                    ax.set_xlabel('y [\si{\metre}]')
                    continue
                #
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                # ax.spines['left'].set_color('none')
                # ax.spines['bottom'].set_color('none')
            plot.annotate('Increasing excitation frequency', xycoords='figure fraction', textcoords='figure fraction', xytext=(0.3, 0.05), xy=(0.9, 0.05), arrowprops=dict(arrowstyle='->', facecolor='black'), verticalalignment='center',)
            plot.annotate('Increasing amplitude', xycoords='figure fraction', textcoords='figure fraction', rotation=90, xytext=(0.05, 0.5), xy=(0.05, 0.9), arrowprops=dict(arrowstyle='->', facecolor='black'), horizontalalignment='center',)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=0.98, hspace=0, wspace=0.03)

            # plot.savefig('/ismhome/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/hysteresis_different_forcing.pdf')
            # plot.savefig('/ismhome/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/hysteresis_different_forcing.png')

            plot.show()


def stepsize_example(ansys):
#########
# Example showing effect of stepsizes
#########
    
    from mpldatacursor import datacursor
    with matplotlib.rc_context(rc=print_context_dict):
        plot.figure(tight_layout=True)
        zeta_ = 0 / 100
        omega_ = 1
        for method, gamma_, beta_, linestyle in [('Average Acceleration', 0.5, 0.25, 'solid'),
                                                     ('Linear Acceleration', 0.5, 1 / 6, 'dashed'),
    #                                                  ('Central Difference',0.5,0,'dotted'),
                                                     ('$\\gamma=0.6, \\beta=0.3025$', 0.6, 0.3025, 'dashdot'),
                                                     ('$\\gamma=0.9142, \\beta=0.5$', np.sqrt(2) - 1 / 2, 1 / 2, (0, (3, 1, 1, 1, 1, 1))), ]:
            generate_sdof_time_hist(ansys, omega=omega_, zeta=zeta_, d0=1, dt_fact=0.1, num_cycles=4, parameter_set=(gamma_, beta_), label=method, linestyle=linestyle)

    #     generate_student(ansys, omega=1, zeta=0, d0=1, dt_fact=0.001, num_cycles=1, parameter_set='LAM', color='grey', label='Analytic')
        t = np.linspace(0, 20, 3000)
        ydata = free_decay(t, R=1 / 2, zeta=zeta_, omega_d=omega_, phi=0)
        plot.plot(t, ydata, color='grey', label='Analytical')
#         plot.plot(t,2*1/2*np.exp(-zeta_*omega_/(np.sqrt(1-zeta_**2))*t), color='grey', label='Envelope', ls='dotted')
#         plot.plot(t,-2*1/2*np.exp(-zeta_*omega_/(np.sqrt(1-zeta_**2))*t), color='grey', label='Envelope', ls='dotted')
        plot.xlim((0, 20))
        plot.ylim((-1.1, 1.1))
        datacursor(formatter='{label}'.format, bbox=None, draggable=True, display='multiple')
    #     plot.legend()
        plot.xlabel('Time [\\si{\\second}]')
        plot.ylabel('Displacement [\\si{\\metre}]')
        plot.show()

#     plot.figure()
#     plot.subplot()
#     import sympy
#     sympy.init_printing()
#     h,beta,gamma,zeta,omega,Omega,eta=sympy.symbols('h \\beta \gamma \zeta \omega \Omega \eta', real=True, positive=True)
#
#     A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[omega**2,2*zeta*omega, 1]])
#     #with this one, Eq. 3.17 gives an oscillating response
#     A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
#     A=A1.solve(A2)
#     dt = 2*np.pi*0.1
#     timesteps =30
#     t = np.linspace(0,stop=dt*timesteps,num=timesteps)
#     for beta_ in [1/6,1/4,0]:
#         Anp = np.array(A.subs(omega,1).subs(zeta,0).subs(h,dt).subs(gamma,0.5).subs(beta,beta_)).astype(np.float64)
#         res = np.zeros((3,timesteps+1))
#         if False:#not beta_:
#             res[0,1]=1 #u0
#             res[0,0]=res[0,1]-dt*res[1,1]+dt**2/2*res[2,1] #u-1
#             res[2,1]=-1 #dot{u}_0
#             k=1
#             m=1
#             khat = m/dt**2
#             a=m/dt**2
#             b=k-2*m/dt**2
#             for i in range(1,timesteps+1):
#                 phat=-a*res[0,i-2]-b*res[0,i-1]
#                 res[0,i] = phat/khat
#             res/=np.pi/2
#         else:
#             res[0,0]=1
#             for i in range(1,timesteps+1):
#
#                 res[:,i]= Anp.dot(res[:,i-1])
#         if False:#not beta_:
#             res = res[:,1:]
#         else:
#             res = res[:,:-1]
#         plot.plot(t,res[0,:], label=f'{beta_}')
#
#     dt = 2*np.pi*0.001
#
#     timesteps =3000
#     t=np.linspace(0,stop=dt*timesteps,num=timesteps)
#     Anp = np.array(A.subs(omega,1).subs(zeta,0).subs(h,dt).subs(gamma,0.25).subs(beta,1/6)).astype(np.float64)
#     res = np.zeros((3,timesteps+1))
#     res[0,0]=1
#     for i in range(1,timesteps+1):
#         res[:,i]= Anp.dot(res[:,i-1])
#     plot.plot(t,res[0,:-1])
#     plot.xlim((0,20))
#     plot.ylim((-1.1,1.1))
#     plot.legend()
#     plot.show()

# def spectr_shift_example(ansys):
#     with matplotlib.rc_context(rc=print_context_dict):
#         fig, axes = plot.subplots(nrows = 2, ncols=1)
#         generate_mdof(ansys, deltat=1/200, num_cycles=8,axes=axes)
#         generate_mdof(ansys, deltat=1/100, num_cycles=8, axes=axes)
#         generate_mdof(ansys, deltat=1/50, num_cycles=8, axes=axes)
#
#
#         axes[1].axvline(-1, color='grey', label="Theoretical")
#
#         axes[0].set_ylabel('Acceleration [\\si{\\metre\\per\\square\\second}]')
#         axes[0].set_xlabel('Time [\\si{\\second}]')
#         axes[1].set_ylabel('PSD [\\si{\\decibel\\per\\hertz}]')
#         axes[1].set_xlabel('Frequency [\\si{\\hertz}]')
#         axes[1].grid(0)
#         axes[1].set_xlim((0,100))
#         axes[1].legend()
#         fig.subplots_adjust(left=0.110, bottom=0.125, right=0.970, top=0.960, hspace=0.340)
#         plot.show()


def  student_data(ansys, ambient=True, nonlinear=True, friction=True):
    # verification study over: zeta 0 %, 1%, 5 %, 20 %, over nonlinearity 0, 0.5, -0.5, over fric_visc_rat 0, 0.5, 1
    # plot.ion()

    omega = np.random.random() * 14 + 1
    zeta = np.random.random() * (10 - 0.1) + 0.1
    dt_fact = np.random.random() * (0.015 - 0.001) + 0.001

    if ambient:
        num_cycles = np.random.randint(300, 2000)  # ambient
        f_scale = np.random.random() * 10
        d0 = None
    else:
        num_cycles = np.random.randint(3, 20)  # free decay
        d0 = np.random.random() * 100
        f_scale = None

    if nonlinear:
        nl_ity = np.random.random() - 0.5
    else:
        nl_ity = 0

    if friction:
        fric_visc_rat = np.random.random()
    else:
        fric_visc_rat = 0

    if ambient and nonlinear and not friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_ambient/'
    elif ambient and not nonlinear and not friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_linear_ambient/'
    elif not ambient and not nonlinear and not friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_linear_decay/'
    elif not ambient and nonlinear and not friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_decay/'
    elif not ambient and not nonlinear and friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_friction_decay/'
    elif ambient and not nonlinear and friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_friction_ambient/'
    elif not ambient and nonlinear and friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_general_decay/'
    elif ambient and nonlinear and friction:
        savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_general_ambient/'
    else:
        raise RuntimeError(f'This combination of inputs is not supported: {ambient}, {nonlinear}, {friction}')

#                 generate_sdof_time_hist(ansys, omega=1, zeta=zeta/100, m=1, fric_visc_rat=fric_visc_rat, nl_ity=nl_ity, d0=1, dt_fact=0.001, num_cycles=20, savefolder=None)
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta / 100, m=1, fric_visc_rat=fric_visc_rat, nl_ity=nl_ity,
                            d0=d0, f_scale=f_scale,
                            dt_fact=dt_fact, num_cycles=num_cycles, savefolder=savefolder)


def student_data_part2(jid, result_dir, omega, zeta, dt_fact, num_cycles, f_scale, d_scale, nl_ity, fric_visc_rat, snr_db, working_dir, **kwargs):
    """

    add noise SNR [-20 dB (S/R = 0.01 Signal zehn mal schwaecher als Rauschen),
                    0 dB (Signalleistung==Rauschleistung),
                   20 dB (S/R=10 Signal zehn mal staerker als Rauschen)]
    reduce sample rate : decimate by factor 6 -> h/T = [0.006 - 0.09] -> samples per cycle = [166 ... 11]  to not mix numerical accuracy effects with each other but rather merge them proportionally

    pre compute covariance up to the total length of time series, even though it is not needed

    Use Data Manager for that for each type feed in the ids and SNR

    define function, that loads, decimates, adds noise and (if ambient: computes correlation function) saves
    the actual definition file can then be updated independently

    """

    global ansys
    try:
        ansys.finish()
    except (pyansys.errors.MapdlExitedError, NameError) as e:
        logger.exception(e)
        ansys = Mechanical.start_ansys(jid=jid, working_dir=working_dir)

    ansys.clear()

    if d_scale is not None:
        if np.isnan(d_scale): d_scale = None

    if f_scale is not None:
        if np.isnan(f_scale): f_scale = None

    array, k, c, d_max, fsl, deltat = generate_sdof_time_hist(ansys,
                                                   omega, zeta / 100, 1, fric_visc_rat, nl_ity ,  # structural parameters
                                                   d_scale, f_scale,  # loading parameters
                                                   dt_fact=dt_fact, num_cycles=num_cycles,  # signal parameters
                                                   savefolder=result_dir, working_dir=working_dir, jid=jid,  # function parameters
                                                   data_hadidi=True, **kwargs)
    plot.plot(array[:,0],array[:,1])
    plot.show()
    snr = 10 ** (snr_db / 10)

    power = np.mean(array[:, 1] ** 2)

    # decimate
    array = array[1::6, :] # why start at 1 here?
    N = array.shape[0]
    # add noise
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), N)
    power_noise = np.mean(noise ** 2)

    snr_actual = power / power_noise
    snr_actual_db = 10 * np.log10(snr_actual)

    array[:, 1] += noise
    if d_scale is None and f_scale is not None:
        cov = np.zeros((N, 2))
        for i in range(N):
            cov[i, 1] = array[:N - i, 1].dot(array[i:, 1]) / (N - i)
        cov[:, 0] = array[:, 0]

        mdict = {'disp':array, 'corr':cov}
    else:
        mdict = {'disp':array}
    file = os.path.join(result_dir, jid + '.mat')
    scipy.io.savemat(file, mdict, appendmat=True, format='5', long_field_names=False, do_compression=True, oned_as='row')

    dt = (array[-1, 0] - array[0, 0]) / (array.shape[0] - 1)

    return k, c, d_max, fsl, power, power_noise, dt


def identify_student():
    source_num = 0

    source_folder = ['/vegas/scratch/womo1998/data_hadidi/datasets/',
                     '/vegas/scratch/womo1998/data_hadidi/datasets_ambient/',
                     '/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_decay/',
                     '/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_ambient/'
                     ][source_num]

    snr = 100
    decimate = 0
    single = False

    id_res = []
    id_inp = []
    fig, axes = plot.subplots(nrows=5, ncols=8)
    axes = axes.flatten()
    # with open('/vegas/users/staff/womo1998/data_hadidi/description_new.txt','tr') as descr:
    with open(f'{source_folder}description.txt', 'tr') as descr:  # , open(f'{source_folder}description_new.txt','tw') as descr_new:
        descr.readline()
        for i, line in enumerate(descr):
            print(line)
            m = 1
            if source_num == 0:
                id, omega, zeta, R, deltat = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(id,k,d,R,deltat))
            elif source_num == 1:
                id, omega, zeta, deltat = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(id,k,d,deltat))
            elif source_num == 2:
                id, omega, zeta, R, deltat, nl_ity = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(id,k,d,R,deltat,nl_ity))
            elif source_num == 3:
                id, omega, zeta, d_max, deltat, nl_ity = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(id,k,d,d_max,deltat,nl_ity))

        # continue

            # print(f'zeta: {zeta*100} \%, omega: {omega} Hz')
            ty = np.loadtxt(f'{source_folder}{id}.csv')
            # print(deltat, ty[2,0]-ty[1,0])
            if snr:
                # SNR=u_eff,sig^2/u_eff,noise^2 (wikipedia: Signal-Rausch-Verhältnis: Rauschspannungsverhältnis)
                # u_eff,noise^2 = u_eff,sig^2/SNR
                ty[:, 1] += np.random.randn(ty.shape[0]) * np.sqrt(ty[:, 1].var() * snr)  # variance equals rms**2 here because it is a zero-mean process

            if decimate:
                ty = ty[::decimate, :]
            if 'ambient' in source_folder:
                if source_num == 3:
                    d = max(np.abs(ty[:, 1]))
                    k *= (1 + nl_ity * ((d / d_max) ** 2 - 1))
                ydata = scipy.signal.correlate(ty[:, 1], ty[:, 1], mode='full', method='direct')
                ydata = ydata[ydata.shape[0] // 2:, ][:1000]
                xdata = ty[:1000, 0]
            else:
                ydata = ty[:, 1]
                xdata = ty[:, 0]
            # popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = ty[:1000,0], ydata=corr[corr.shape[0]//2:,][:1000], p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
            try:
                popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata=xdata, ydata=ydata, p0=[0.5, 0.05, 2 * np.pi, 0])  # ,  bounds=[(-1,0,0,0),(1,1,np.pi/deltat,2*np.pi)])
            except Exception as e:
                print('ID failed', e)
                popt = [0, 0, 0, 0]
                pcov = [np.inf, np.inf, np.inf, np.inf]

            perr = np.sqrt(np.diag(pcov))
            # print(' zeta: {:1.3f}+-{:1.3f} \%, omega: {:1.4f}+-{:1.3f} Hz\n\n'.format( popt[1]*100, perr[1]*100, popt[2], perr[2]))
            id_res.append((popt[2], popt[1], xdata[2] - xdata[1]))

            t_synth = np.linspace(0, xdata[-1], 1000)
            # synth=free_decay(ty[:1000,0], *popt)
            synth = free_decay(t_synth, *popt)

            zeta = zeta
            omega = np.sqrt(k / m) * np.sqrt(1 - zeta ** 2)

            deltat = deltat
            id_inp.append((omega, zeta, deltat))
            print(omega, zeta, deltat)

            if single:
                plot.figure()
                plot.plot(xdata, ydata, ls='none', marker=',')
                plot.plot(t_synth, synth)
                plot.show()
                continue
            # axes[i].plot(ty[:1000,0],corr[corr.shape[0]//2:,][:1000], ls='none', marker='+')
            axes[i].plot(xdata, ydata, ls='none', marker=',')
            axes[i].plot(t_synth, synth)
#             if len(id_res)>2:
#                 break
            # if i >4:break
    id_res = np.array(id_res)
    id_inp = np.array(id_inp)
    labels = ['Omega', 'Zeta', 'Delta_t']
    for i in range(2):
        plot.figure()
        plot.plot(id_inp[:, i], id_res[:, i], ls='none', marker='.')
        plot.xlabel(labels[i] + '_simulated')
        plot.ylabel(labels[i] + '_identified')

    plot.show()


def generate_mdof_time_hist(ansys, num_nodes=None, damping=None, nl_stiff=None, sl_forces=None, freq_scale=1, num_modes=None,  # structural parameters
                            d0=None, f_scale=None,  # loading parameters
                            deltat=None, dt_fact=None, timesteps=None, num_cycles=None, num_meas_nodes=None, meas_nodes=None,  # signal parameters
                            savefolder=None, working_dir='/dev/shm/womo1998/', jid=None,  # function parameters
                            ** kwargs):
    '''
    argument:
        damping = None -> no damping
                  zeta -> global rayleigh,
                  (zeta, zeta) -> global rayleigh
                  (alpha,beta,True/False) -> global/elementwise proportional
                  (eta, True/False) -> Structural/Hysteretic (global -> only damped modal and harmonic analyses not TRANSIENT! / elementwise -> only transient (IWAN))
                  (Friction -> only local via sl_forces
                  IWAN needs an equivalent damping ratio and a stress-strain relationship (G,E?)
                    from which a monotonic strain-softening backbone curve \tau vs \gamma (considering shear) according to Masing's rule can be computed (Kausel Eq. 7.307),
                    in turn this can be approximated/discretized by N spring-slider systems in parallel (or a nonlinear stiffness
                    TRY IN SDOF first
        d0 = initial displacement
                'static' : displacement obtained from static analysis with unit top displacement
                'modes' : arithmetic mean of all mode shapes
                number > 0, modenumber of which


    TODO:
    - spring/mass elements and assembly
        - validation in static and modal analyses ✓
        - number of nodes needed for consideration of a certain number of modes (convergence study and plot) ✓
    -  viscous damping elements and assembly
        - validation with global (rayleigh damping) in stiffness matrices and modal analysis  ✓
        - automatic rayleigh parameters with analytical solution before ✓
        - comparison of global and element damping in terms of modal damping (numerical) vs number of nodes for x modes and y target damping ratios ✓

    - free decay
        - transient analysis parameters (timestep/dt-omega-ratio, duration/num_cycles, numerical damping if not all modes are required?)
            - set parameters for e.g. ten modes and check the response
        - identification of modal parameters from pure mode free decay and mixed free decay (filtered?) -> verification
    - ambient
        - uncorrelated random processes / correlated random processes
        - identification of modal parameters using OMA -> verification

    MAYBE STOP HERE
    - nonlinear-equivalent
        - stiffness (as in SDOF systems), estimation of d_max
        - friction (equivalent per node and mode, averaging?)
        - structural damping (ANSYS routines, IWAN)
    '''
    try:
        ansys.finish()
    except pyansys.errors.MapdlExitedError as e:
        print(e)
        ansys = Mechanical.start_ansys()
    ansys.clear()
    
    mech = Mechanical(ansys=ansys, jobname=jid, wdir=working_dir)
    mech.example_rod(num_nodes, damping, nl_stiff, sl_forces, freq_scale, num_modes, num_meas_nodes, meas_nodes)

    if kwargs.pop('just_build', False):
        return mech
    
    if kwargs.pop('convergence_study', False):
        disp = mech.static(uz=np.array([[num_nodes, 1]]))
        frequencies, damping, modeshapes = mech.modal(damped=False, num_modes=num_modes)
        return [coords[3] for coords in mech.nodes_coordinates], disp[:, 2], frequencies, modeshapes[:, 2, :].real

    if kwargs.pop('damping_study', False):
        frequencies, damping, modeshapes = mech.modal(damped=True, num_modes=num_modes)
        return frequencies, damping, mech.alpha, mech.beta

    if kwargs.pop('evd_study', False):
        frequencies, damping, modeshapes = mech.modal(damped=True, num_modes=num_modes)
        return mech, frequencies, damping, modeshapes, mech.alpha, mech.beta

    if kwargs.pop('verify', False):

        if d0 is not None:
            t_vals, resp_hist = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles, **kwargs)
        elif f_scale is not None:
            t_vals, resp_hist, inp_hist = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles, **kwargs)

#         resp_hist=resp_hist[0] #accelerations

        frequencies, damping, modeshapes = mech.numerical_response_parameters(compensate=False)
        frequencies_n, damping_n, modeshapes_n = mech.numerical_response_parameters()

        timesteps = mech.timesteps
        mode = 0

        mech.ansys.save(f'{mech.jobname}.db')
        mech.ansys.exit()
        meas_nodes = mech.meas_nodes

        if False:
            from core import PreprocessingTools
            from core import PlotMSH

            mech.export_geometry(f'/dev/shm/womo1998/{jid}/')
            geometry = PreprocessingTools.GeometryProcessor.load_geometry(f'/dev/shm/womo1998/{jid}/grid.txt', f'/dev/shm/womo1998/{jid}/lines.txt')
            prep_data = PreprocessingTools.PreprocessData(resp_hist, 1 / (t_vals[1] - t_vals[0]))
            chan_dofs = prep_data.load_chan_dofs(f'/dev/shm/womo1998/{jid}/chan_dofs.txt')
            prep_data.add_chan_dofs(chan_dofs)

            mode_shape_plot = PlotMSH.ModeShapePlot(geometry, prep_data=prep_data)
            mode_shape_plot.draw_nodes()
            mode_shape_plot.draw_lines()
            mode_shape_plot.draw_master_slaves()
            mode_shape_plot.draw_chan_dofs()
            PlotMSH.start_msh_gui(mode_shape_plot)
        elif True:

            nrows = int(np.ceil(np.sqrt(len(meas_nodes))))
            ncols = int(np.ceil(len(meas_nodes) / nrows))
            fig, axes = plot.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
            fig2, axes2 = plot.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)

            for channel, (ax, ax2, meas_node) in enumerate(zip(axes.flat, axes2.flat, meas_nodes)):
                try:
                    popt, _ = response_frequency(t_vals, resp_hist[0][:, channel], p0=[1, damping[mode], frequencies[mode] * 2 * np.pi, 0])
                    resid_id = np.sqrt(np.mean((resp_hist[0][:, -1] - free_decay(t_vals, *popt)) ** 2))
                    print(resid_id)
                except Exception as e:
                    print(e)
                    popt = [1, 1, 0, 0]
                frequencies_id = popt[2] / 2 / np.pi
                damping_id = popt[1]
                time_fine = np.arange(deltat, deltat * timesteps, 0.0001)

                for i in range(1):
                    ax.plot(t_vals, resp_hist[i][:, channel], marker='x')
                    ax2.psd(resp_hist[i][:, channel], NFFT=25600, Fs=1 / (t_vals[1] - t_vals[0]), label=str(meas_node))

                ax.plot(time_fine, free_decay(time_fine, 1, damping[mode], frequencies[mode] * 2 * np.pi, 0), label='ex')
                ax.plot(time_fine, free_decay(time_fine, 1, damping_n, frequencies_n * 2 * np.pi, 0), label='est')
                ax.plot(time_fine, free_decay(time_fine, *popt), label='id')

                ax.legend(loc=1)
                ax2.legend(loc=1)
            plot.show(block=True)
            print(frequencies_id, damping_id)
        return

    if kwargs.pop('perf_bench', False):
        if np.isnan(d0): d0 = None
        else: d0 = int(d0)
        if np.isnan(f_scale): f_scale = None

        if d0 is not None:
            t_vals, resp_hist = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles, **kwargs)
        elif f_scale is not None:
            t_vals, resp_hist, inp_hist = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles, **kwargs)

        dts = mech.transient_runtimes
        return np.mean(dts), np.min(dts), np.max(dts), np.std(dts)

    if kwargs.pop('verify_compens', False):
        '''
        verification of numerical compensation:
        Arguments:
        nodes_coordinates    -    a linear SDOF system should be generated with
        k_vals, m_vals       -    natural circular frequency should ideally be omega=1 (disp, vel and acc are all unity)
        damping              -    viscous damping can be applied according to the usual procedures
        meth, parameter_set  -    all available time integration algorithms can be used with different parameters

        returns a range of 20 timestepsizes where numerical errors should be present
        for each timestepsize:
            free_decay with n_cycles and curve fit for identification
        return exact, compensated and identified frequencies and damping ratios

        final analyses to be done outside this function

        '''
        freq_ex, damp_ex, _ = mech.modal(damped=True)
        freq_ex = freq_ex[0]
        damp_ex = damp_ex[0]

        mech.free_decay(d0=0, dscale=1, deltat=0.0001, num_cycles=num_cycles, timesteps=2, **kwargs)
        frequencies_n, damping_n, _ = mech.numerical_response_parameters()

        frequencies = np.ones((20,)) * freq_ex
        damping = np.ones((20,)) * damp_ex

        frequencies_id = np.zeros((20,))
        damping_id = np.zeros((20,))
        frequencies_n = np.zeros((20,))
        damping_n = np.zeros((20,))
        resid_id = np.zeros((20,))

        _, axes = plot.subplots(4, 5, sharex=True, sharey=True)

        deltats = np.logspace(-3, -2, 20) * 3 / freq_ex

        for i, (ax, deltat) in enumerate(zip(axes.flat, deltats)):

            t_vals, resp_hist = mech.free_decay(d0=0, dscale=1, deltat=deltat, num_cycles=num_cycles, timesteps=timesteps, **kwargs)
            freq, damp, modeshape = mech.numerical_response_parameters()

            ax.plot(t_vals, resp_hist[0][:, -1], ls='none', marker='x')
            ax.plot(t_vals, resp_hist[1][:, -1], ls='none', marker='x')
            ax.plot(t_vals, resp_hist[2][:, -1], ls='none', marker='x')
            frequencies_n[i] = freq[0]
            damping_n[i] = damp[0]

            y0 = np.real((modeshape[:] / modeshape[-1]))[0]

            t_vals_fine = np.linspace(0, t_vals[-1], 2048)
            ax.plot(t_vals_fine, free_decay(t_vals_fine, y0[0], damp[0], freq[0] * 2 * np.pi, 0), label='comp')
            try:

                popt, _ = response_frequency(t_vals, resp_hist[0][:, -1], p0=[y0[0], damp[0], freq[0] * 2 * np.pi, 0])  # R, zeta, omega_d, phi=0
                resid_id[i] = np.sqrt(np.mean((resp_hist[0][:, -1] - free_decay(t_vals, *popt)) ** 2))
                print(resid_id[i])
                frequencies_id[i] = popt[2] / 2 / np.pi
                damping_id[i] = popt[1]
            except Exception as e:
                popt = [0, 0, 0, 0]
                print(e)
            ax.plot(t_vals_fine, free_decay(t_vals_fine, *popt), label=f'id f={popt[2]/2/np.pi:.5f} \t d={popt[1]:.5f}')
            ax.legend()

        ax.set_ylim((-4, 4))

        return frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id, resid_id

    if kwargs.pop('verify_impulse', False):
        dt_fact, deltat, num_cycles, timesteps, _ = mech.signal_parameters(dt_fact, deltat, num_cycles, timesteps)

        ref_nodes = [2, 3]
        imp_durs = [20 * deltat, 40 * deltat]

        imp_times = [1 * deltat, 51 * deltat]
        imp_forces = [10000, 10000]

#         ref_nodes = [2]
#         imp_durs = [ 10*deltat]
#
#         imp_times = [ 20*deltat]
#         imp_forces=[ 10000]

#         for form in ['step','rect','sine']:
#             t_vals, resp_hist, inp_hist= mech.impulse_response([ref_nodes,imp_forces,imp_times,imp_durs], form=form, mode='combined', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles,**kwargs)
        ref_nodes = range(2, num_nodes + 1)
        imp_forces = [196, 56, 29, 19, 14, 11, 9, 8, 8]
        imp_times = [deltat for i in range(num_nodes)]
        imp_durs = [30 * deltat for i in range(num_nodes)]
        out_quant = ['d']
        t_vals, IRF_matrix, F_matrix, ener_mat, amp_mat = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form='step', mode='matrix', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles, out_quant=out_quant, **kwargs)

        num_ref_nodes = len(ref_nodes)
        meas_nodes = mech.meas_nodes
        num_meas_nodes = len(meas_nodes)
        jid = ansys.jobname

        mech.export_geometry(f'/dev/shm/womo1998/{jid}/')
        from core import PreprocessingTools
        geometry = PreprocessingTools.GeometryProcessor.load_geometry(f'/dev/shm/womo1998/{jid}/grid.txt', f'/dev/shm/womo1998/{jid}/lines.txt')

        if out_quant[0] == 'd':
            disp_channels = list(range(num_meas_nodes))
            velo_channels = None
            accel_channels = None
        elif out_quant[0] == 'a':
            accel_channels = list(range(num_meas_nodes))
            velo_channels = None
            disp_channels = None
        elif out_quant[0] == 'v':
            velo_channels = list(range(num_meas_nodes))
            disp_channels = None
            accel_channels = None
        channel_headers = meas_nodes
        ref_channels = [i for i, node in enumerate(meas_nodes) if node in ref_nodes]
        dummy_meas = np.zeros((timesteps, num_meas_nodes))

        print(accel_channels, velo_channels, disp_channels, ref_channels,)

        prep_data = PreprocessingTools.PreprocessData(dummy_meas, 1 / deltat, ref_channels=ref_channels,
                                                      accel_channels=accel_channels,
                                                      velo_channels=velo_channels,
                                                      disp_channels=disp_channels,
                                                      channel_headers=channel_headers)
        chan_dofs = prep_data.load_chan_dofs(f'/dev/shm/womo1998/{jid}/chan_dofs.txt')
        prep_data.add_chan_dofs(chan_dofs)
        prep_data.save_state(f'/dev/shm/womo1998/{jid}/prep_data.npz')
        np.savez(f'/dev/shm/womo1998/{jid}/IRF_data.npz', t_vals=t_vals, IRF_matrix=IRF_matrix, F_matrix=F_matrix, ener_mat=ener_mat, amp_mat=amp_mat)

        # IRF Matrix
        # geometry
        # ref_channels, accel_channels, velo_channels, disp_channels

        return
    
    

    if d0 is not None:
        t_vals, resp_hist = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles, **kwargs)
    elif f_scale is not None:
        t_vals, resp_hist, inp_hist = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles, **kwargs)
    resp_hist = resp_hist[2]  # accelerations

    frequencies, damping, modeshapes = mech.numerical_response_parameters()

    for file in glob.glob(f'{working_dir}{jid}.*'):
        # print(f'removing old files: {file}')
        os.remove(file)

    # for i, node in enumerate(mech.meas_nodes):
        # ydata = resp_hist[:, i]
        # plot.plot(t_vals, ydata, label=f'{node}')
    # plot.legend()
    # plot.show()

    if savefolder is not None:
        mech.save(savefolder)
        mech.export_geometry(savefolder)
    
    return mech

# def mean_confidence_interval(data, confidence=0.95):  # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    # a = 1.0 * np.array(data)
    # n = len(a)
    # m, se = np.mean(a), scipy.stats.sem(a)
    # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    # return m, h


def convergence_mdof_undamped(ansys):
    mshs = []
    freqs = []
    coords = []
    omega_true = [(2 * k - 1) / 2 * np.pi / 200 * np.sqrt(2.1e11 / 7850) for k in range(1, 11)]
    num_modes = 10

    x = []
    if False:
        for num_nodes in range(2, 21 ** 2):
            print(num_nodes)
            coord, disp, frq, msh = generate_mdof_time_hist(ansys, num_nodes, num_modes=num_modes, convergence_study=True)
            freqs.append(frq)
            mshs.append(msh)
            coords.append(coord)
            # x.append(num_nodes)

            # freq_errors.append(np.mean(np.abs(frq-frq_true[:len(frq)])/frq_true[:len(frq)]))

            # msh_true = np.array([np.sin(np.array(coords)*(2*i-1)/2*np.pi/-200) for i in range(1,len(frq)+1)]).T
            # msh_errors.append(np.abs(np.abs(msh)-np.abs(msh_true)).flatten())

        np.savez('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/example_convergence_msh.npz', msh_errors=mshs, freq_errors=freqs, coords=coords)
        return
    else:
        arr = np.load('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/example_convergence_msh.npz', allow_pickle=True)
        mshs = arr['msh_errors']
        freqs = arr['freq_errors']
        coords = arr['coords']
        x = list(range(2, 21 ** 2))
    with matplotlib.rc_context(rc=print_context_dict):
        fig, (ax2, ax1) = plot.subplots(2, 1, sharex=True, squeeze=True)
        for mode, color, marker in zip([8, 4, 1], ['lightgrey', 'darkgrey', 'black'], ['+', 'x', '.']):
            freq_errors = []
            msh_errors = []
            this_x = []
            for j, num_nodes in enumerate(x):
                if num_nodes < 9: continue
                if j % 5: continue  # every fifth point only
                this_x.append(num_nodes)
                # print(num_nodes)
                omega = freqs[j]
                coord = coords[j]
                msh = mshs[j]
                freq_errors.append(np.mean(np.abs(omega[mode - 1] * 2 * np.pi - omega_true[mode - 1]) / omega_true[mode - 1]) * 100)
                # print(omega[mode-1]*2*np.pi, omega_true[mode-1])
                msh_true = np.sin(np.array(coord) * (2 * mode - 1) / 2 * np.pi / -200).T
                # print(msh.shape, msh_true.shape)
                msh_errors.append(np.mean(np.abs(np.abs(msh[:, mode - 1]) - np.abs(msh_true))))

            ax1.plot(this_x, freq_errors, ls='none', marker=marker, color=color, label='$\\epsilon(\\omega_' + str(mode) + ')$')
            ax2.plot(this_x, [np.mean(err) for err in msh_errors], ls='none', marker=marker, color=color, label='$\\epsilon(\\bm{\\phi}_' + str(mode) + ')$')

        # ax1.set_xlabel('\#nodes')
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('$n$')
        ax1.set_ylabel('$\\nicefrac{\\abs{\\omega_\\text{num}-\\omega_\\text{ex}}}{\\omega_\\text{ex}}$ [\\si{\\percent}]')
        ax2.set_ylabel('$\\overline{\\abs{\\bm{\\phi}_\\text{num}-\\bm{\\phi}_\\text{ex}}}$ [\\si{\\metre}]')
        ax1.set_xlim((0, 400))
        ax2.set_xlim((0, 400))
        ax1.set_ylim((0, 0.05 * 100))
        # ax1.set_yscale('log')
        ax2.set_ylim((0, 1e-13))

        plot.subplots_adjust(left=0.11, right=0.970, top=0.95, bottom=0.115, hspace=0.1)
        plot.savefig('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/example_convergence_mdof.pdf')
        plot.savefig('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/example_convergence_mdof.png')
        plot.show()


def rayleigh_example(ansys):
    '''
    Study constant modal damping over frequency (rayleigh, stiffness-, mass-proportional):
    - Analytical
    - Numerical global
    - Numerical local


    '''
    plot.figure()
    ax1 = plot.subplot()
    plot.figure()
    ax2 = plot.subplot()
    for color, zeta in zip(['lightgrey', 'black'], [0.025, 0.1]):

        num_nodes = 80

        freqs, damps, alpha, beta = generate_mdof_time_hist(ansys, num_nodes, damping=(zeta, zeta), num_modes=8, damping_study=True)
        # print(freqs)
        fs = np.linspace(0, 100, 1000)
        omegas = fs * 2 * np.pi
        zetas_r = 1 / 2 * (alpha / omegas + beta * omegas) * 100
        zetas_m = 1 / 2 * (alpha / omegas + 0 * omegas) * 100
        zetas_k = 1 / 2 * (0 / omegas + beta * omegas) * 100

        if zeta == 0.1:
            ax1.plot(fs, zetas_r, color=color, ls='solid', label=f'Rayleigh')
            ax1.plot(fs, zetas_m, color=color, ls='dashdot', label=f'Stiffness prop.')
            ax1.plot(fs, zetas_k, color=color, ls='dotted', label=f'Mass prop.')
        else:
            ax1.plot(fs, zetas_r, color=color, ls='solid')
            ax1.plot(fs, zetas_m, color=color, ls='dashdot')
            ax1.plot(fs, zetas_k, color=color, ls='dotted')
        # ax1.plot([freqs[0],freqs[-1]],[200*zeta,100*zeta], color=color, ls='dashed',lw=0.75)#target damping

        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (alpha / omegas + beta * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='x', color=color, label='Rayleigh (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color, label='Rayleigh (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', marker='x', color=color)
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color)

        # stiffness proportional
        freqs, damps, _, _ = generate_mdof_time_hist(ansys, num_nodes, damping=(0, beta, True), num_modes=8, damping_study=True)
        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (0 / omegas + beta * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='+', color=color, label='Stiffness prop. (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color, label='Stiffness prop. (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', marker='+', color=color)
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color)

        # mass proportional
        freqs, damps, _, _ = generate_mdof_time_hist(ansys, num_nodes, damping=(alpha, 0, True), num_modes=8, damping_study=True)
        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (alpha / omegas + 0 * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='1', color=color, label='Mass prop. (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='1', color=color, label='Mass prop. (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', marker='1', color=color)
            ax2.plot(freqs, delta_damps, ls='none', marker='1', color=color)

        # Elementwise stiffness proportional
        freqs, damps, _, _ = generate_mdof_time_hist(ansys, num_nodes, damping=(0, beta, False), num_modes=8, damping_study=True)
        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (0 / omegas + beta * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='3', color=color, label='Elementwise (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='3', color=color, label='Elementwise (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', color=color, marker='3')
            ax2.plot(freqs, delta_damps, ls='none', color=color, marker='3')

    ax1.legend()
    ax1.set_xlim((0, 100))
    ax1.set_ylim((0, 200 * zeta))
    ax1.set_xlabel('Natural frequency $f \\, [\\si{\\hertz}]$')
    ax1.set_ylabel('Damping ratio  $\\zeta \\, [\\si{\\percent}_\\text{crit}]$')

    ax2.legend()
    ax2.set_xlim((0, 100))

    plot.sca(ax1)
    plot.subplots_adjust(top=0.95, bottom=0.125, left=0.11, right=0.960)
    # plot.savefig('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/example_rayleigh_damping.pdf')
    # plot.savefig('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/example_rayleigh_damping.png')

    plot.show()


def eigenvalue_decomposition_example(ansys):

    omega_d = np.pi * 2
    freq_scale = omega_d * np.sqrt(200 ** 2 * 7850 / 2 / 2.1e11)
    freq_scale=1
    np.set_printoptions(precision=2, linewidth=250)
    num_nodes = 11
    num_modes = num_nodes - 1
    import scipy.linalg
    mech, freqs, damps, mshs, alpha, beta = generate_mdof_time_hist(ansys, freq_scale=freq_scale, num_nodes=num_nodes, damping=None, num_modes=num_modes, evd_study=True)
    print('ANSYS undamped modal parameters')
    print(freqs, '\n', damps)

    k, m, c = mech.export_ans_mats()
    print(k,m,c)
    full_path = os.path.join(ansys.directory, ansys.jobname + '.full')
    full = pyansys.read_binary(full_path)
    # TODO: Check, that Nodes and DOFS are in the same order in modeshapes and k,m
    dof_ref, k_, m_ = full.load_km(as_sparse=False, sort=False)
    k_ += np.triu(k_, 1).T
    m_ += np.triu(m_, 1).T
#     print(dof_ref)
    for mode in range(num_modes):
        msh_f = mshs[1:, 2, mode].flatten()

        kappa = msh_f.T.dot(k).dot(msh_f)
        mu = msh_f.T.dot(m).dot(msh_f)
        
        print(np.sqrt(kappa / mu) / 2 / np.pi)

        msh_f = mshs[:, :, mode].flatten()

        kappa = msh_f.T.dot(k_).dot(msh_f)
        mu = msh_f.T.dot(m_).dot(msh_f)
        print(np.sqrt(kappa / mu) / 2 / np.pi)

    # print('ANSYS undamped system matrices')
    # print(k,'\n',m,'\n',c)
    w, vr = scipy.linalg.eig(k, m)
    print('SCIPY modal frequencies')
    print(np.sqrt(w) / 2 / np.pi)

    mech, freqs, damps, mshs, alpha, beta = generate_mdof_time_hist(ansys, freq_scale=freq_scale, num_nodes=num_nodes, damping=0.01, num_modes=num_modes, evd_study=True)
    print(alpha, beta)
    print('ANSYS damped modal parameters')
    print(freqs, '\n', damps)
    a0, a2, a1 = mech.export_ans_mats()
    
    k, m, c = mech.export_ans_mats()
    for mode in range(num_modes):
        msh_f = mshs[1:, 2, mode].flatten()

        kappa = msh_f.T.dot(k).dot(msh_f)
        mu = msh_f.T.dot(m).dot(msh_f)
        eta = msh_f.T.dot(c).dot(msh_f)
        print(mode, kappa.real, mu.real, eta.real)
    # a1*=-1
    print('ANSYS damped system matrices')
    print(a0, '\n', a2, '\n', a1)
    zero = np.zeros_like(a0)
    factor = 1e8
    eye = factor * np.eye(*a0.shape)
    A = np.hstack((np.vstack((zero, -eye)), np.vstack((a0, a1))))  # golub p 415
    B = factor * np.eye(*A.shape)
    B[-(a2.shape[0]):, -(a2.shape[1]):] = a2
    print('Linearized system matrices')
    print(A, '\n', B)
    w, vr = scipy.linalg.eig(A, -B)
    print('SCIPY eigenvalues and eigenvectors')
    print(w, '\n', vr)
    print('SCIPY modal parameters')
    omega0 = np.abs(w)  # brinckers p 102
    zeta = -np.real(w) / omega0
    omegad = np.imag(w)
    phi = vr[num_modes:, ::2]
    modmass = np.diag((phi.T).dot(a2).dot(phi))
    phi /= np.sqrt(modmass)
    print(type(phi))
    print(omega0 / 2 / np.pi, '\n', omegad / 2 / np.pi, '\n', zeta)
    for i in range(num_modes):
        color = plot.plot(np.hstack(([0], phi[:, i].real)), label=omega0[i * 2] / 2 / np.pi)[0].get_color()
        plot.plot(np.hstack((mshs[:, 2, i].real)), color=color, ls='none', marker='x')
        # print([f"{a:.3e}" for a in np.real(phi[:, i])])
        
        kappa = phi[:, i].T.dot(k).dot(phi[:, i])
        mu = phi[:, i].T.dot(m).dot(phi[:, i])
        eta = phi[:, i].T.dot(c).dot(phi[:, i])
        print(mode, omega0[i * 2] / 2 / np.pi, kappa.real, mu.real, eta.real)
    plot.legend()
    plot.show()



# def start_ansys(working_dir=None, jid=None,):
    #
    # # global ansys
    # # try:
        # # ansys.finish()
    # # except (pyansys.errors.MapdlExitedError, NameError) as e:
        # # logger.warning(repr(e))
    # if working_dir is None:
        # working_dir = os.getcwd()
    # os.chdir(working_dir)
    # now = time.time()
    # if jid is None:
        # jid = 'file'
    # ansys = pyansys.launch_mapdl(
        # exec_file='/vegas/apps/ansys/v201/ansys/bin/ansys201',
        # run_location=working_dir, override=True, loglevel='WARNING',
        # nproc=1, log_apdl='w',
        # log_broadcast=False, jobname=jid,
        # mode='console', additional_switches='-smp')
        #
    # logger.debug(f'Took {time.time()-now} s to start up ANSYS.')
    #
    # ansys.clear()
    #
    # return ansys


def model_performance():
    from uncertainty.data_manager import DataManager

    if False:
        data_manager = DataManager(title='model_perf2', working_dir='/dev/shm/womo1998/')

        num_nodes = np.array([5, 10, 15, 25, 35, 45, 60, 75, 90, 110, 130, 150, 175, 200, 225, 250])
        rat_meas_nodes = np.random.random(size=num_nodes.size)
        chunksize = np.array([500, 1000, 2000, 4000])
        NUM_NODES, RAT_MEAS_NODES, CHUNKSIZE = [array.flatten() for array in np.meshgrid(num_nodes, rat_meas_nodes, chunksize, indexing='ij')]

        D0 = np.zeros((NUM_NODES.size), dtype=float)
        D0[::3] = 1
        F_SCALE = np.logical_not(D0) * 100000000.0
        F_SCALE[F_SCALE == 0] = np.nan
        D0[D0 == 0] = np.nan

        NUM_MEAS_NODES = np.floor(NUM_NODES * RAT_MEAS_NODES)
        NUM_MEAS_NODES[NUM_MEAS_NODES < 2] = 2

        arrays = [NUM_NODES, D0, F_SCALE, NUM_MEAS_NODES, CHUNKSIZE]
        names = ['num_nodes', 'd0', 'f_scale', 'num_meas_nodes', 'chunksize']

        data_manager.provide_sample_inputs(arrays, names)
        return
    elif False:
        data_manager = DataManager.from_existing(dbfile_in='model_perf2.nc', result_dir='/usr/scratch4/sima9999/work/modal_uq/')
        # data_manager.clear_failed(True)
        data_manager.post_process_samples()

        return
    else:
        ansys = Mechanical.start_ansys()

        data_manager = DataManager.from_existing(dbfile_in='model_perf2.nc', result_dir='/usr/scratch4/sima9999/work/modal_uq/')
        func_kwargs = {'ansys':ansys, 'damping':0.05, 'dt_fact':0.01, 'timesteps':4001, 'perf_bench':True}
        arg_vars = [('num_nodes', 'num_nodes'), ('d0', 'd0'), ('f_scale', 'f_scale'), ('num_meas_nodes', 'num_meas_nodes'), ('chunksize', 'chunksize')]
#         data_manager.evaluate_samples(func=generate_mdof_time_hist, arg_vars=arg_vars, ret_names=['mean', 'min', 'max', 'std'],**func_kwargs,
#                                       chwdir=False
#                                       )
        data_manager.evaluate_samples(func=generate_mdof_time_hist, arg_vars=arg_vars, ret_names=['num_meas_nodes'], **func_kwargs,
                                      chwdir=False
                                      )

        return


def verify_numerical_accuracy(ansys, m=None, d=1, r=1):
    '''
    check the effect of nsubst, autots, deltim on the accuracy
    read again ansys manual, especially with respect to alpha methods, look for further parameters
    current settings allow full compensation for newmark
    but give equal deviations for all alpha methods
    period elongations are smaller than expected for g-alpha and wbz and lager than expected for hht
    numerical damping is larger than expected for all alpha methods
    '''

    # accuracy_study(ansys) # precursor

    damp = [None, 0.1][d]

    df = 0.05
    # rho = 0.5 euqals df=1/3
    # rho= 1/3 is the last value, that ansys does not complain about
    rho = [1, (-df + 1) / (df + 1), 0.5, 1 / 3 ][r]
    parameter_sets = ['AAM', 'LAM', rho, rho, rho, rho]
    meths = ['NMK', 'NMK', 'NMK', 'HHT', 'WBZ', 'G-alpha']

    if m is not None:  # build data

        meth = meths[m]
        parameter_set = parameter_sets[m]
        omega_d = 2  # *np.pi
        freq_scale = omega_d * np.sqrt(200 ** 2 * 7850 / 2 / 2.1e11)

        if damp is not None:
            freq_scale /= np.sqrt(1 - damp ** 2)  # generate_mdof does not correct for damping, so we have to correct our frequency beforehand

#         freq_scale*=np.pi
#         frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id = generate_mdof_time_hist(ansys, damping=None, num_nodes=2, freq_scale=freq_scale,
#                                    deltat=0.01, num_cycles=800, meas_nodes=[2],
#                                    verify=True, d0=0,
#                                    meth=meth, parameter_set=parameter_set
#                                    )
#         return
#

        arrs = generate_mdof_time_hist(ansys, damping=damp, num_nodes=2, freq_scale=freq_scale,  # d_0=1, a_0=4, v_pi=2
                                       num_cycles=24, meas_nodes=[2],
                                       verify_compens=True, meth=meth, parameter_set=parameter_set)
        frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id, resid_id = arrs

        np.savez(file=f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/verify_compens_m{m}_r{r}_d{d}.npz',
             frequencies=frequencies,
             frequencies_n=frequencies_n,
             frequencies_id=frequencies_id,
             damping=damping,
             damping_n=damping_n,
             damping_id=damping_id,
             resid_id=resid_id)
        plot.suptitle(f'{meth}')
        plot.tight_layout()
        figManager = plot.get_current_fig_manager()
        figManager.window.showMaximized()
        plot.show()
        return
    else:
        linestyles = ["solid", "solid", 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
        markers = ['3', '4', '+', 'x', '1', '2']
        meths = ['AAM', 'LAM', 'NMK', 'HHT-$\\alpha$', 'WBZ-$\\alpha$', 'G-$\\alpha$']
        fig, axes = plot.subplots(2, 4, sharey=True, gridspec_kw={'width_ratios':(3, 1, 3, 1)})
#         fig2,ax2 = plot.subplots(1,1)
        for d in reversed(range(2)):
            color = ['black', 'gray'][d]
            damp = [None, 0.1][d]

            for m in range(2, 6):
                meth = meths[m]
                print(f'verify_compens_m{m}_r{r}_d{d}', os.path.getmtime(f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/verify_compens_m{m}_r{r}_d{d}.npz'))
                arrs = np.load(f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/verify_compens_m{m}_r{r}_d{d}.npz')
                frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id, resid_id = [value[:20] for value in arrs.values()]

                periods_id = 1 / frequencies_id
                periods_n = 1 / frequencies_n
                periods = 1 / frequencies

                h = np.logspace(-3, -2, 20) * 3 / frequencies
                h = h[:20]

#                 ax2.plot(resid_id, h/periods, label=f'm{m}-d{d}')

                axes[d, 0].semilogy(periods_n, h / periods, ls=linestyles[m], color=color)
                axes[d, 0].semilogy(periods_id, h / periods, color=color, marker=markers[m], ls='none')
                axes[d, 2].semilogy(damping_n, h / periods, color=color, ls=linestyles[m])
                axes[d, 2].semilogy(damping_id, h / periods, color=color, marker=markers[m], ls='none')
                axes[d, 1].semilogy(np.abs(periods_id - periods_n) / periods * 100, h / periods, color=color, marker=markers[m], ls='none')
                axes[d, 3].semilogy(np.abs(damping_id - damping_n), h / periods, color=color, marker=markers[m], ls='none')

            axes[d, 1].set_xlim((-0.0004, 0.049))
            axes[d, 3].set_xlim((-1e-7, 2.5e-5))
            axes[d, 1].xaxis.set_ticklabels(['-2.5e-2', '0', '2.5e-2', '5e-2'])
            axes[d, 3].xaxis.set_ticklabels(['-2.5e-5', '0', '2.5e-5'])
        axes[1, 0].set_xlabel('$T [\si{\second}]$')
        axes[1, 1].set_xlabel('$|\Delta T|$ [\si{\percent}]')
        axes[1, 2].set_xlabel('$\zeta$ [\si{\percent}]')
        axes[1, 3].set_xlabel('$|\Delta \zeta|$ [\si{\percent}]')

        axes[0, 0].yaxis.set_major_formatter(plot.NullFormatter())
#         axes[0,0].yaxis.set_minor_formatter(LogFormatterSciNotation())
        for d in range(2):
            axes[d, 0].set_ylabel('$\sfrac{h}{T}$')
            axes[d, 0].yaxis.set_label_coords(-0.2, 0.55)
            axes[d, 0].set_ylim((0.003, 0.03))
            axes[d, 0].set_xlim((3.1415, 3.1549))

        leg_handles = []
        for label, ls, mkr in list(zip(meths, linestyles, markers))[2:]:
            line = matplotlib.lines.Line2D([], [], color='black', marker=mkr, ls=ls, label=label)
            leg_handles.append(line)
        axes[0, 0].legend(handles=leg_handles).set_draggable(True)
        fig.subplots_adjust(top=0.970, bottom=0.115, left=0.112, right=0.970, hspace=0.180, wspace=0.1)
        axes[1, 2].set_xlim(xmax=0.1039)
#         plot.suptitle(f'$\\rho = {rho}$')
#         plot.savefig('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/example_accuracy_timeint.pdf')
#         plot.savefig('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/modeling_realization/example_accuracy_timeint.png')
#         ax2.legend()
        plot.show()
        return


def IRF_to_ssi(ansys=None, jid=None, **kwargs):

    from core.SSICovRef import BRSSICovRef

    from core.PreprocessingTools import PreprocessData, GeometryProcessor

    # Modal Analysis PostProcessing Class e.g. Stabilization Diagram
    from core.StabilDiagram import StabilCalc, StabilPlot
    from GUI.StabilGUI import StabilGUI, start_stabil_gui

    # Modeshape Plot
    from core.PlotMSH import ModeShapePlot
    from GUI.PlotMSHGUI import start_msh_gui

    if ansys is not None:
        omega_d = np.pi / 2
        freq_scale = omega_d * np.sqrt(200 ** 2 * 7850 / 2 / 2.1e11)
        # freq_scale/=np.sqrt(1-0.05**2)
        num_nodes = 10
        num_cycles = 15,
        deltat = 0.005
        mech = generate_mdof_time_hist(ansys, damping=0.01, num_nodes=num_nodes, num_modes=num_nodes - 1,  # freq_scale=freq_scale,
                           just_build=True)

        dt_fact, deltat, num_cycles, timesteps, _ = mech.signal_parameters(deltat=deltat, num_cycles=num_cycles)

        ref_nodes = range(2, num_nodes + 1)
        imp_forces = [196, 56, 29, 19, 14, 11, 9, 8, 8]
        imp_times = [deltat for i in range(num_nodes)]
        imp_durs = [30 * deltat for i in range(num_nodes)]
        out_quant = ['a']
        t_vals, IRF_matrix, F_matrix, ener_mat, amp_mat = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form='step', mode='matrix', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles, out_quant=out_quant, **kwargs)

        frequencies, damping, modeshapes = mech.numerical_response_parameters()
        meas_nodes = mech.meas_nodes
        num_meas_nodes = len(meas_nodes)

        jid = ansys.jobname

        mech.export_geometry(f'/dev/shm/womo1998/{jid}/')
        import PreprocessingTools

        if out_quant[0] == 'd':
            disp_channels = list(range(num_meas_nodes))
            velo_channels = None
            accel_channels = None
        elif out_quant[0] == 'a':
            accel_channels = list(range(num_meas_nodes))
            velo_channels = None
            disp_channels = None
        elif out_quant[0] == 'v':
            velo_channels = list(range(num_meas_nodes))
            disp_channels = None
            accel_channels = None
        channel_headers = meas_nodes
        ref_channels = [i for i, node in enumerate(meas_nodes) if node in ref_nodes]
        dummy_meas = np.zeros((timesteps, num_meas_nodes))

        prep_data = PreprocessingTools.PreprocessData(dummy_meas, 1 / deltat, ref_channels=ref_channels,
                                                      accel_channels=accel_channels,
                                                      velo_channels=velo_channels,
                                                      disp_channels=disp_channels,
                                                      channel_headers=channel_headers)
        chan_dofs = prep_data.load_chan_dofs(f'/dev/shm/womo1998/{jid}/chan_dofs.txt')
        prep_data.add_chan_dofs(chan_dofs)
        prep_data.save_state(f'/dev/shm/womo1998/{jid}/prep_data.npz')
        np.savez(f'/dev/shm/womo1998/{jid}/IRF_data.npz', t_vals=t_vals, IRF_matrix=IRF_matrix, F_matrix=F_matrix, ener_mat=ener_mat, amp_mat=amp_mat, frequencies=frequencies, damping=damping, modeshapes=modeshapes)

    else:
        assert jid is not None

    # creating the geometry for plotting the identified modeshapes
    geometry_data = GeometryProcessor.load_geometry(f'/dev/shm/womo1998/{jid}/grid.txt', f'/dev/shm/womo1998/{jid}/lines.txt')

    prep_data = PreprocessData.load_state(f'/dev/shm/womo1998/{jid}/prep_data.npz')

    arrs = np.load(f'/dev/shm/womo1998/{jid}/IRF_data.npz')
    t_vals = arrs['t_vals']
    IRF_matrix = arrs['IRF_matrix']
    F_matrix = arrs['F_matrix']
    ener_mat = arrs['ener_mat']
    amp_mat = arrs['amp_mat']
    frequencies = arrs['frequencies']
    damping = arrs['damping']
    modeshapes = arrs['modeshapes']

    print(frequencies, damping, modeshapes)

    tau_max = IRF_matrix.shape[2]
    prep_data.tau_max = tau_max
    prep_data.corr_matrix = IRF_matrix
    prep_data.corr_matrices = [IRF_matrix]

    modal_data = BRSSICovRef(prep_data)
    modal_data.build_toeplitz_cov(num_block_columns=125)
    modal_data.compute_state_matrices(max_model_order=19)

    frequencies_id, damping_id, modeshapes_id, _, modal_contributions = modal_data.single_order_modal(order=18)

    sort_inds = np.argsort(frequencies_id[:9])
    frequencies_id = frequencies_id[sort_inds]
    damping_id = damping_id[sort_inds]
    modal_contributions = modal_contributions[sort_inds]
    modeshapes_id = modeshapes_id[:, sort_inds]

    for mode in range(9):
        modeshapes_id[:, mode] /= modeshapes_id[np.argmax(np.abs(modeshapes_id[:, mode])), mode]
        modeshapes[:, mode] /= modeshapes[np.argmax(np.abs(modeshapes[:, mode])), mode]

        modeshapes_id[:, mode] /= modeshapes_id[-1, mode]
        modeshapes[:, mode] /= modeshapes[-1, mode]

    plot.figure()
    plot.plot(frequencies, ls='none', marker='+')
    plot.plot(frequencies_id, ls='none', marker='x')
    plot.figure()
    plot.plot(damping * 100, ls='none', marker='+')
    plot.plot(damping_id, ls='none', marker='x')
    plot.figure()
    plot.plot(modeshapes)
    plot.plot(modeshapes_id, ls='dotted')
    plot.figure()
    total_energy = np.sum(ener_mat)
    plot.plot(frequencies, np.sum(ener_mat, axis=0) / total_energy, ls='none', marker='+')
    plot.plot(frequencies_id, modal_contributions, ls='none', marker='x')
    plot.show()

    if False:
        modal_data.compute_modal_params()

        stabil_data = StabilCalc(modal_data)
        stabil_plot = StabilPlot(stabil_data)
        start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_data)

        mode_shape_plot = ModeShapePlot(geometry_data, stabil_data, modal_data, prep_data)
        start_msh_gui(mode_shape_plot)


def test():
    '''
    A function to test basic functionality of the Mechanical class
    a lot of things are not yet verified/validated
    but at least the code runs, mistakes will be found later, maybe...
    '''
    
    working_dir = '/dev/shm/womo1998/'
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)
    
    save_dir = '/usr/scratch4/sima9999/work/modal_uq/test_mechanical/'
    
    jid = 'test_mechanical'
    ansys = Mechanical.start_ansys(working_dir, jid)
    
    if True:
        # Example structure Burscheid Longitudinal modes
        total_height = 200
        E = 2.1e11
        # I=0.01416
        A = 0.0343
        rho = 7850
    
        num_nodes = 41
        num_meas_nodes = 10
    
        section_length = total_height / (num_nodes - 1)
        nodes_coordinates = []
        k_vals = [0 for _ in range(num_nodes - 1)]
        masses = [0 for _ in range(num_nodes)]
        d_vals = None  # [0 for i in range(num_nodes-1)]
        eps_duff_vals = [0 for _ in range(num_nodes - 1)]
        sl_force_vals = [0 for _ in range(num_nodes - 1)]
        hyst_vals = [0 for _ in range(num_nodes - 1)]
        
        damping = (0.01, 0.02)
        
        for i in range(num_nodes):
            nodes_coordinates.append([i + 1, 0, 0, 0])  # to disable Warning "Nodes are not coincident"
            masses[i] += 0.5 * rho * A * section_length
            if i >= 2:
                masses[i - 1] += 0.5 * rho * A * section_length
            if i >= 1:
                k_vals[i - 1] = E * A / section_length
                
                # if nl_stiff is not None:
                    # eps_duff_vals[i-1]=nl_stiff
                # if sl_forces is not None:
                    # sl_force_vals[i-1]=sl_forces
                # if hyst_damp is not None:
                    # hyst_vals [i-1] = hyst_damp
    
        meas_nodes = np.rint(np.linspace(1, num_nodes, int(num_meas_nodes + 1))).astype(int)
    
        mech = Mechanical(ansys, jid, wdir=os.getcwd())
        mech.build_mdof(nodes_coordinates=nodes_coordinates,
                        k_vals=k_vals, masses=masses, d_vals=d_vals, damping=damping,
                        sl_force_vals=sl_force_vals, eps_duff_vals=eps_duff_vals,hyst_vals=hyst_vals,
                        meas_nodes=meas_nodes)
        N = 2**16
        fmax = None
        out_quant = 'a'
        inp_node = -1
        mech.frequency_response(N, fmax, out_quant, inp_node)
        
        mech.save(save_dir)
        del mech
        
    mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    if not mech.state[1]:
        d0 = 1
        deltat = None
        dt_fact = 0.01
        timesteps = None
        num_cycles = 20
        t_vals_dec, resp_hist_dec = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles)
        
        mech.save(save_dir)
        mech.del_test()
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    if False:#not mech.state[2]:
        f_scale = 1000
        deltat = 1/512
        dt_fact = None
        timesteps = 60*512
        num_cycles = None
        t_vals_amb, resp_hist_amb, inp_hist_amb = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles)
        
        mech.save(save_dir)
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    if not mech.state[3]:
        deltat = 1/512
        dt_fact = None
        timesteps = 6*512
        num_cycles = None
        ref_nodes = mech.meas_nodes[-5:]
        imp_forces = [100 for _ in ref_nodes]
        imp_times = [deltat for _ in ref_nodes]
        imp_durs = [30 * deltat for _ in ref_nodes]
        out_quant = ['d']
        t_vals_imp, IRF_matrix, F_matrix, ener_mat, amp_mat = \
            mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs],
                                  form='step', mode='matrix', deltat=deltat,
                                  dt_fact=dt_fact, timesteps=timesteps,
                                  num_cycles=num_cycles, out_quant=out_quant)
        
        mech.save(save_dir)
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
        
    if not mech.state[5]:
        frequencies, damping, modeshapes = mech.numerical_response_parameters()
        
        mech.save(save_dir)
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    


def main():
    
    global ansys
    
    try:
        ansys.finish()
    except (pyansys.errors.MapdlExitedError, NameError) as e:
        logger.exception(e)
        global working_dir
        working_dir = '/dev/shm/womo1998/'
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)
        global jid
        jid = str(uuid.uuid4()).split('-')[-1]
        ansys = Mechanical.start_ansys(working_dir, jid)

    ansys.clear()
#         IRF_to_ssi(ansys)
#     else:
#         IRF_to_ssi(jid='5a208083e5a1')
#     return
    # global working_dir
    # working_dir = os.getcwd()
    # working_dir = '/usr/scratch4/sima9999/work/modal_uq/'
    # working_dir = '/dev/shm/womo1998/'
    # os.makedirs(working_dir, exist_ok=True)
    # os.chdir(working_dir)

#     import glob
#     filelist = glob.glob('/dev/shm/womo1998/*')
#     for file in filelist:
#         os.remove(file)

    # identify_student()

    # global jid
    # global ansys
    # jid=str(uuid.uuid4()).split('-')[-1]

    '''


    '''


#
    # nodes_coordinates = [(1, 0, 0, 0), (2, 0, 0, 0)]
    # k_vals = [1]
    # d_vals = None
    # damping = 0
    # meas_nodes = [2]
    # num_modes = 1
    #
    # omegas = np.linspace(.1, 10, 25, True)
    # fix, axes = plot.subplots(5, 5, sharex=True, sharey=True)
    # energies = []
    # for omega, ax in zip(omegas, axes.flat):
        # mech = Mechanical(ansys, jid, wdir=os.getcwd())
        # masses = [0, k_vals[0] / omega ** 2]
        #
        # mech.build_mdof(nodes_coordinates=nodes_coordinates,
                    # k_vals=k_vals, masses=masses, d_vals=d_vals, damping=damping,
                    # meas_nodes=meas_nodes, num_modes=num_modes)
                    #
        # deltat = 0.0025
        # timesteps = 801
        #
        # ref_nodes = [2]
        # imp_durs = [1]
        #
        # imp_times = [1 * deltat]
        # imp_forces = [1000]
        #
        # form = "rect"
        #
        # time_values, response_time_history, f , modal_imp_energies, modal_amplitudes = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form=form, mode='combined', deltat=deltat, timesteps=timesteps)
        # # print(modal_imp_energies)
        # energies.append(modal_imp_energies[1])
        # ax.plot(time_values, response_time_history[0], label=f'{omega}')
        # ax.axvline(1)
        # ax.set_title(f' $\omega={omega:.2f}$')
    # for ax in axes[-1, :]:
        # ax.set_xlabel("t [s]")
    # for ax in axes[:, 0]:
        # ax.set_ylabel("d [m]")
        #
    # fig, ax = plot.subplots(1, 1)
    # ax.plot(omegas, energies)
    # plot.show()

#     import sys
#     num=int(sys.argv[1])-1
#     m = num%6
#     d=num//6%2
#     r=num//12
# #     ## m 0...5, d = 0..1, r = 0...3
#     verify_numerical_accuracy(ansys, m, d, r)
#     verify_numerical_accuracy(ansys, m=5, d=0, r=0)
#     with matplotlib.rc_context(rc=print_context_dict):
#         verify_numerical_accuracy(ansys, r=1)

#     with matplotlib.rc_context(rc=print_context_dict):
#         rayleigh_example(ansys)

    eigenvalue_decomposition_example(ansys)

#     convergence_mdof_undamped(ansys)

#     import sys
#     # sys.argv = [..., ambient, nonlinear, friction]
#     if len(sys.argv)>3:
#         ambient = int(float(sys.argv[1]))
#         nonlinear = int(float(sys.argv[2]))
#         friction = int(float(sys.argv[3]))
#     else:
#         ambient=False
#         nonlinear=False
#         friction=True
#     student_data(ansys, ambient, nonlinear, friction)

#     hysteresis(ansys)

#     stepsize_example(ansys)


if __name__ == '__main__':

    '''
    Static response
        Nonlinear Stiffness
    Quasi-Static
        Nonlinear Stiffness
        Friction Response
    Modal Analysis
    Dynamic Response (Nyquist Check, Amplitude Check)
       Initial Displacement -> Free Vibration
           Numerical Damping
           Prescribed Damping
       Harmonic Loading
           Nonlinear Stiffness
           Damping Hysteresis
       Impulse
           IRF
       Sweep / Random Loading
           FRF

    MDOF
    Pass all verifications
    Random Field Loading
    Mesh Size
    Meas Nodes Definition
    Full Parametrization
    '''
#     main()

    """
    Test
    decay,ambient
        undamped
        damped
            linear, friction, nonlinear, friction+nonlinear

    """
    test()
    #main()
#     for d_scale, f_scale in [(1, None), (None, 1)]:
#         #if d_scale: continue
#         for zeta in [0, 1.2151028666644974]:
#             #if not zeta: continue
#             for fric_visc_rat, nl_ity in [(0, 0), (1, 0), (0, .5), (1, .5), ]:
# #                 if not fric_visc_rat: continue
# 
#                 if d_scale:
#                     jid = "decay_"
#                 else:
#                     jid = "ambient_"
#                 if zeta:
#                     jid += "damped_"
#                 if nl_ity:
#                     jid += "nonlinear_"
#                 if fric_visc_rat:
#                     jid += "friction_"
# 
#                 student_data_part2(jid=jid, result_dir='/vegas/scratch/womo1998/test/',
#                            omega=2*np.pi*1.7240461713677018, zeta=zeta,
#                            dt_fact=0.005562, num_cycles=1304,
#                            f_scale=f_scale, d_scale=d_scale,
#                            nl_ity=nl_ity, fric_visc_rat=fric_visc_rat,
#                            snr_db=np.infty, working_dir='/dev/shm/womo1998/')

