import sys
import os
sys.path.append("/home/sima9999/code/")
sys.path.append("/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/")

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

from model.turbulent_wind import terrain_parameters, basic_wind_parameters, spectral_wind_field,temporal_wind_field,force_wind_field,plot_windfield
from model.mechanical import Mechanical, MechanicalDummy
from model.acquisition import Acquire, sensor_position

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.SSICovRef import BRSSICovRef
from pyOMA.core.SSIData import SSIDataCV
from pyOMA.core.PLSCF import PLSCF
from pyOMA.core.StabilDiagram import StabilCalc
global ansys

from uncertainty.polymorphic_uncertainty import MassFunction, RandomVariable, PolyUQ

def stage2n3mapping(n_locations,
                    DTC, 
                    sensitivity_nominal, sensitivity_deviation_percent, 
                    spectral_noise_slope, sensor_noise_rms,
                    range_estimation_duration, range_estimation_margin,
                    DAQ_noise_rms,
                    decimation_factor, anti_aliasing_cutoff_factor,
                    quant_bit_factor,
                    duration,
                    m_lags, estimator, model_order,
                    jid, result_dir, working_dir, skip_existing=True, **kwargs):
        
    modules = ['pyOMA.core.Helpers','pyOMA.core.PreProcessingTools','pyOMA.core.SSICovRef','pyOMA.core.SSIData','pyOMA.core.PLSCF','model.mechanical','model.acquisition']
    for module in modules:
        logger_ = logging.getLogger(module)
        logger_.setLevel(logging.WARNING)
        
    bits_effective, snr_db_est, snr_db, channel_defs, acqui = stage2mapping(n_locations,
                            DTC, 
                            sensitivity_nominal, sensitivity_deviation_percent, 
                            spectral_noise_slope, sensor_noise_rms,
                            range_estimation_duration, range_estimation_margin,
                            DAQ_noise_rms,
                            decimation_factor, anti_aliasing_cutoff_factor,
                            quant_bit_factor,
                            duration, 
                            jid=jid, 
                            result_dir=result_dir, working_dir = working_dir, skip_existing=skip_existing, 
                            chained_mapping=True)
    
    estimator =  ['blackman-tukey','welch'][estimator]
    
    f_sc, d_sc, phi_sc, mc_sc, \
    f_cf, d_cf, phi_cf, mc_cf, \
    f_sd, d_sd, phi_sd, mc_sd = _stage3mapping(m_lags=m_lags, estimator=estimator,
                            n_blocks=40, k=10, model_order=model_order, 
                            jid=jid, 
                            result_dir=result_dir, working_dir=working_dir, skip_existing=skip_existing, 
                            acqui_obj=acqui)
    # create an phi_indexer to re-order modeshapes
    # assumes, phi_*'s last row to contain np.nan: phi_*[n_locations * 2, :] = np.nan
    # apply it as phi_*[phi_indexer,:] to get sparse mode shapes in the order of the model's mode shapes
    
    '''
    nodes_coordinates = [(n1,x1,y1,z1), (n2,x2,y2,z2)]
    mech.mode_shapes  = [ [(qx1,qy,qz), (qx2,q2y,qz2), ...]_1,
                          [(qx1,qy,qz), (qx2,q2y,qz2), ...]_2, ]
    acqui.mode_shapes = [ [qa,qb,qc,...]_1,
                          [qa,qb,qc,...]_2, ]
    oma.mode_shapes   = [ [qa,qb,qc,...]_1,
                          [qa,qb,qc,...]_2, ]
    channel_defs: maps [(qx1,qy1,qz1), (qx2,qy2,qz2),...] to [qa,qb,...]
    phi_indexer: maps [qa,qb,...] to [(qx1,qy1,qz1), (qx2,qy2,qz2),...]
    
    so phi_indexer is the reverse map of channel_defs
    
    for mode comparison mode shapes must be flattened
    flattening is achieved by reshaping (specifying index order?) and then removing all-nan-rows
    '''
    # print(channel_defs)
    n_nodes = 203
    # phi indexer will lead to modeshapes in y and z coordinate only
    # to make it compatible with mech.mode_shapes, the x-coordinate has to be removed there
    phi_indexer = np.full((n_nodes,2), n_locations * 2, dtype=int)
    for chan, (node, dir_, _) in enumerate(channel_defs):
        node_index = node - 1 # actually, that should be looked up form mech.nodes_coordinates, but should work here
        phi_indexer[node_index,dir_] = chan
    # to save storage, instead of converting all mode shapes, only the phi_indexer is stored for every sample

    return bits_effective, snr_db_est, snr_db,\
            f_sc, d_sc, phi_sc, mc_sc, \
            f_cf, d_cf, phi_cf, mc_cf, \
            f_sd, d_sd, phi_sd, mc_sd, \
            phi_indexer
    

def stage3mapping(m_lags, estimator, model_order,
                  jid, result_dir, working_dir, skip_existing=True, **kwargs):
    # return with default: n_blocks=40, k = 10
    
    estimator =  ['blackman-tukey','welch'][estimator]
    return _stage3mapping(m_lags=m_lags, estimator=estimator,
                          n_blocks=40, k=10, model_order=model_order, 
                          jid=jid, result_dir=result_dir, 
                          working_dir=working_dir, skip_existing=skip_existing, **kwargs)

def _stage3mapping(m_lags, estimator,
                  n_blocks, k, 
                  model_order, 
                  jid, result_dir, working_dir, skip_existing=True, **kwargs):
    # logger = logging.getLogger(__name__)
    # logger.setLevel(level=logging.INFO)
    
    '''
    p + q = m_lags          -> num_block_rows = m_lags // 2
    p = q = m_lags // 2     -> num_block_rows = m_lags // 2
    n_{+} = m_lags // 2 + 1 -> nperseg = m_lags
    
    
    What are the bounds on m_lags, order, etc. ?
    
    fixed parameters:
        n_blocks = 40
        n_r = 2
        
    parameters from acqui:
        channel_headers -> n_l
        sampling_rate 
        signals -> N = shape[0]
        
    
    corr_blackman_tukey:
        N_segment = N // n_blocks
                  = N // 40
        N_segment > m_lags
    
    corr_welch:
        N_segment = N // n_segments
        n_lines = (m_lags - 1) * 2
        
        (bypassed in psd_welch: N_segment < n_lines)
        (avoid zero-padding: N_segment > n_lines / 2)
    
    SSICovRef:
        num_block_rows = num_block_columns = m_lags // 2
        K = min(n_l * (num_block_rows + 1), n_r * num_block_columns)
        (here specifically)
          = n_r * num_block_columns
          = n_r * (m_lags // 2)
        
        order <= K
              <= 2 * m_lags //2
              <= m_lags
    
    pLSCF:
        nperseg = m_lags
        
        order < nperseg - 1 
              <= m_lags - 2
    
    SSIDataCV:
        q = p = num_block_rows = m_lags // 2
        N_b = (N - q - p) / n_blocks
            = (N - m_lags) / 40
        N_b > n_r * q
            > m_lags
            
        K = n_r * q
          = 2 * m_lags // 2
          = m_lags
          
        order < K
              < m_lags
    
    ======================================
    m_lags < N // 40 # corr_blackman_tukey
    m_lags < N // 41 # SSIData
    
    order < m_lags # SSICovRef
    order < m_lags - 2 # pLSCF
    order < m_lags # SSIData
    
    Final bound to be checked:
    order + 2 < m_lags < N // 41
    =====================================
    
    Returns: f_sc, f_sc, phi_sc, mc_sc: order modal parameter from SSICovRef (only order//2 are present)
            f_c,f f_cf, phi_cf, mc_cf: n_r * order modal parameter from pLSCF (typically four times as many modes)
            f_sd, f_sd, phi_sd, mc_sd: order modal parameter from SSIDataCV (only order//2 are present)
    '''
    
    assert model_order + 2 < m_lags
    
    if not isinstance(result_dir, Path):
        result_dir = Path(result_dir)
    
    if not isinstance(working_dir, Path):
        working_dir = Path(working_dir)
    
    # Set up directories
    id_ale, id_epi = jid.split('_')
    this_result_dir = result_dir / id_ale
    this_result_dir = this_result_dir / id_epi
    seed = int.from_bytes(bytes(id_ale, 'utf-8'), 'big')
    assert os.path.exists(this_result_dir) 

    prep_signals = None
    if os.path.exists(this_result_dir / 'prep_signals.npz') and skip_existing:
        try:
            prep_signals = PreProcessSignals.load_state(this_result_dir / 'prep_signals.npz')
        except Exception as e:
            # os.remove(this_result_dir / 'measurement.npz')
            print(e)
            
    if prep_signals is None:
        if 'acqui_obj' in kwargs:
            acqui = kwargs['acqui_obj']
        else:
            assert os.path.exists(this_result_dir / 'measurement.npz')        
            acqui = Acquire.load(this_result_dir / 'measurement.npz', differential='sampled')

        pd_kwargs = acqui.to_prep_data()
        ref_channels=np.where(acqui.channel_defs[:,0]==201)[0]

        assert ref_channels.shape[0] == 2

        N = pd_kwargs['signals'].shape[0]
        if m_lags > N // (n_blocks + 1):
            raise RuntimeError(f"m_lags > N // (n_blocks + 1): {m_lags} > {N // (n_blocks + 1)}")

        del acqui
           
    #if prep_signals is None:
        prep_signals = PreProcessSignals(**pd_kwargs, ref_channels=ref_channels)
        # prep_signals.corr_blackman_tukey(m_lags, num_blocks=n_blocks, refs_only=True)
        if estimator == 'blackman-tukey':
            prep_signals.corr_blackman_tukey(m_lags, num_blocks=n_blocks)
        elif estimator == 'welch':
            prep_signals.corr_welch(m_lags, n_segments=n_blocks)
        prep_signals.save_state(this_result_dir / 'prep_signals.npz')
    
    
    
    rng = np.random.default_rng(seed)
    cardinality = n_blocks // k
    block_indices = np.arange(cardinality*k)
    rng.shuffle(block_indices)
    
    # holdout method, single run with a training and test set 
    i = rng.integers(0,k)
    
    test_set = block_indices[i * cardinality:(i + 1) * cardinality]
    training_set = np.take(block_indices, np.arange((i + 1) * cardinality, (i + k) * cardinality), mode='wrap')    
    
    if estimator == 'blackman-tukey':
        training_corr = np.mean(prep_signals.corr_matrices_bt[training_set,...], axis=0)
        test_corr = np.mean(prep_signals.corr_matrices_bt[test_set,...], axis=0)    
    elif estimator == 'welch':
        training_corr = np.mean(prep_signals.corr_matrices_wl[training_set,...], axis=0)
        test_corr = np.mean(prep_signals.corr_matrices_wl[test_set,...], axis=0)    

    ssi_cov_ref = None
    if os.path.exists(this_result_dir / 'ssi_cov_ref.npz') and skip_existing:
        try:
            ssi_cov_ref = BRSSICovRef.load_state(this_result_dir / 'ssi_cov_ref.npz', prep_signals)
        except Exception as e:
            # os.remove(this_result_dir / 'measurement.npz')
            logger.warning(repr(e))
            
    if ssi_cov_ref is None:
        ssi_cov_ref = BRSSICovRef(prep_signals)

        if estimator == 'blackman-tukey':
            prep_signals.corr_matrix_bt = training_corr
        elif estimator == 'welch':
            prep_signals.corr_matrix_wl = training_corr
            
        ssi_cov_ref.build_toeplitz_cov() # expensive, should be saved
        # ssi_cov_ref.save_state(this_result_dir / 'ssi_cov_ref.npz')
        
    if estimator == 'blackman-tukey':
        prep_signals.corr_matrix_bt = test_corr
    elif estimator == 'welch':
        prep_signals.corr_matrix_wl = test_corr
        
    A_sc, C_sc, G_sc = ssi_cov_ref.estimate_state(model_order)
    f_sc, d_sc, phi_sc, lamda_sc = ssi_cov_ref.modal_analysis(A_sc, C_sc)
    _, mc_sc = ssi_cov_ref.synthesize_correlation(A_sc, C_sc, G_sc)  # expensive, last step in analysis: does not have to be saved

    del ssi_cov_ref
    
    plscf = None
    if os.path.exists(this_result_dir / 'plscf.npz') and skip_existing:
        try:
            plscf = PLSCF.load_state(this_result_dir / 'plscf.npz', prep_signals)
        except Exception as e:
            # os.remove(this_result_dir / 'plscf.npz')
            logger.warning(repr(e))
            
    if plscf is None:
        plscf = PLSCF(prep_signals)
        
        if estimator == 'blackman-tukey':
            prep_signals.corr_matrix_bt = training_corr
        elif estimator == 'welch':
            prep_signals.corr_matrix_wl = training_corr
            
        plscf.build_half_spectra() # must not compute psds
        # plscf.save_state(this_result_dir / 'plscf.npz')

    if estimator == 'blackman-tukey':
        prep_signals.corr_matrix_bt = test_corr
    elif estimator == 'welch':
        prep_signals.corr_matrix_wl = test_corr
        
    alpha, beta_l_i = plscf.estimate_model(model_order)  # expensive, but not assigning class variables that could be saved
    f_cf, d_cf, phi_cf, lamda_cf = plscf.modal_analysis_residuals(alpha, beta_l_i)
    _, mc_cf = plscf.synthesize_spectrum(alpha, beta_l_i, modal=True)  # expensive, but not assigning class variables that could be saved

    del plscf

    del training_corr
    del test_corr
    prep_signals.corr_matrices_bt = None
    prep_signals.corr_matrices_wl = None
    
    ssi_data = None
    if os.path.exists(this_result_dir / 'ssi_data.npz') and skip_existing:
        try:
            ssi_data = SSIDataCV.load_state(this_result_dir / 'ssi_data.npz', prep_signals)
        except Exception as e:
            # os.remove(this_result_dir / 'ssi_data.npz')
            logger.warning(repr(e))
            
    if ssi_data is None:
        ssi_data = SSIDataCV(prep_signals)
        ssi_data.build_block_hankel(num_block_rows=m_lags // 2, num_blocks=n_blocks, training_blocks=training_set)  # expensive 

        # ssi_data.save_state(this_result_dir / 'ssi_data.npz')
        
    A_sd,C_sd,Q_sd,R_sd,S_sd = ssi_data.estimate_state(model_order)
    f_sd, d_sd, phi_sd, lamda_sd, = ssi_data.modal_analysis(A_sd, C_sd)
    _, mc_sd = ssi_data.synthesize_signals( A_sd, C_sd, Q_sd, R_sd, S_sd, test_set)  # expensive, but not assigning class variables that could be saved

    del ssi_data
    
    return f_sc, d_sc, phi_sc, mc_sc, f_cf, d_cf, phi_cf, mc_cf, f_sd, d_sd, phi_sd, mc_sd, 

def stage2mapping(n_locations,
            DTC, 
            sensitivity_nominal, sensitivity_deviation_percent, 
            spectral_noise_slope, sensor_noise_rms,
            range_estimation_duration, range_estimation_margin,
            DAQ_noise_rms,
            decimation_factor, anti_aliasing_cutoff_factor,
            quant_bit_factor,
            duration,
            jid, result_dir, working_dir, skip_existing=True, **kwargs):
    
    if not isinstance(result_dir, Path):
        result_dir = Path(result_dir)
    
    if not isinstance(working_dir, Path):
        working_dir = Path(working_dir)
    
    # Set up directories
    if '_' in jid:
        id_ale, id_epi = jid.split('_')
        this_result_dir_ale = result_dir / id_ale
        assert os.path.exists(this_result_dir_ale)
        seed = int.from_bytes(bytes(id_ale, 'utf-8'), 'big')
        # print(seed)
        this_result_dir = this_result_dir_ale / id_epi
        
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
    else:
        this_result_dir = result_dir / jid
        this_result_dir_ale = this_result_dir
        assert os.path.exists(this_result_dir)
        seed = int.from_bytes(bytes(jid, 'utf-8'), 'big')
    
    # num_nodes = mech.num_nodes
    num_nodes = 203
    
    setups = sensor_position(n_locations, num_nodes, 'distributed')
    # select a setup based on a "random" integer modulo the total number of setups
    i_setup = seed % setups.shape[0]
    sensor_nodes = setups[i_setup,:]
    quant = 'a'
    quant = ['d', 'v', 'a'].index(quant)
    # list of (node, dof, quant)
    channel_defs = []
    for node in sensor_nodes:
        # We work around the Hack from transient_ifrf, where we omitted the x-axis in the response,
        # by wrongly definining dofs ux and uy instead of uy and uz
        
        for dof in ['ux','uy']:
            dof = ['ux', 'uy', 'uz'].index(dof)
            channel_defs.append((node, dof, quant))

    
    if os.path.exists(this_result_dir / 'measurement.npz') and skip_existing:
        try:
            acqui = Acquire.load(this_result_dir / 'measurement.npz', differential='nosigs')
        except Exception as e:
            # os.remove(this_result_dir / 'measurement.npz')
            print(e)
    else:
        # logger= logging.getLogger('model.acquisition')
        # logger.setLevel(level=logging.INFO)
        if not os.path.exists(this_result_dir_ale / 'response.npz'):
            raise RuntimeError(f"Response does not exist at {this_result_dir_ale / 'response.npz'}")
        assert os.path.exists(result_dir / 'mechanical.npz') 

        mech = MechanicalDummy.load(fpath=result_dir / 'mechanical.npz')

        arr = np.load(this_result_dir_ale / 'response.npz')
        
        # Here's the Hack
        mech.t_vals_amb = arr['t_vals']
        mech.resp_hist_amb = [arr['d_freq_time'], arr['v_freq_time'], arr['a_freq_time']]
        mech.deltat = mech.t_vals_amb[1] - mech.t_vals_amb[0]
        mech.timesteps = mech.t_vals_amb.shape[0]
        mech.state[2] = True

        
        # the above hack conflicts with the modeshape dof though, 
        # so we have to hack around it as well
        mech.damped_mode_shapes = np.delete(mech.damped_mode_shapes, (0), axis=1)

        acqui = Acquire.init_from_mech(mech, channel_defs)
        if logger.isEnabledFor(logging.DEBUG):
            fig, axes = plt.subplots(n_locations, 2, sharex=True, sharey=True)
            t_vals, signal = acqui.get_signal() 
            for i_location in range(n_locations):
                axes[i_location,0].plot(t_vals, signal[i_location * 2,:], alpha=0.5)
                axes[i_location,1].plot(t_vals, signal[i_location * 2 + 1,:], alpha=0.5)
        
        sensitivity_deviation = sensitivity_deviation_percent / 100 * sensitivity_nominal
        acqui.apply_sensor(DTC=DTC, 
                             sensitivity_nominal=sensitivity_nominal, sensitivity_deviation=sensitivity_deviation, 
                             spectral_noise_slope=spectral_noise_slope, noise_rms=sensor_noise_rms)
        logger.debug(acqui.snr)
        
        if logger.isEnabledFor(logging.DEBUG):
            t_vals, signal = acqui.get_signal() 
            for i_location in range(n_locations):
                axes[i_location,0].plot(t_vals, signal[i_location * 2,:], alpha=0.5)
                axes[i_location,1].plot(t_vals, signal[i_location * 2 + 1,:], alpha=0.5)
                
        meas_range = acqui.estimate_meas_range(sample_dur=range_estimation_duration, margin=range_estimation_margin)
        
        quantization_bits = quant_bit_factor * 4
        anti_aliasing_cutoff = anti_aliasing_cutoff_factor * acqui.sampling_rate / decimation_factor
        acqui.sample(dec_fact=decimation_factor, aa_cutoff=anti_aliasing_cutoff, 
                       bits=quantization_bits, meas_range=meas_range, 
                       duration=duration)
        # add noise here, because sampling (decimation) removes all noise again
        logger.debug(acqui.snr)
        
        acqui.add_noise(noise_power=DAQ_noise_rms**2)
        
        logger.debug(acqui.snr)
        if logger.isEnabledFor(logging.DEBUG):
            t_vals, signal = acqui.get_signal() 
            for i_location in range(n_locations):
                axes[i_location,0].plot(t_vals, signal[i_location * 2,:], alpha=0.5)
                axes[i_location,1].plot(t_vals, signal[i_location * 2 + 1,:], alpha=0.5)
        
            
        acqui.estimate_snr()
        
        # acqui.save(this_result_dir / 'measurement.npz', differential='sampled')
        acqui.save(this_result_dir / 'measurement.npz', differential='nosigs')
    
    if kwargs.get('chained_mapping', False):
        return np.array(acqui.bits_effective), np.array(acqui.snr_db_est), np.array(np.mean(acqui.snr_db)), channel_defs, acqui,
    
    return np.array(acqui.bits_effective), np.array(acqui.snr_db_est), np.array(np.mean(acqui.snr_db)), channel_defs
    
    
    

# Spatial domain grid
# x_grid = np.arange(1,201,1)

def default_mapping(zeta=0.002, Iy = 0.01196, Iz = 0.01304, alpha=45, 
            v_b=25.0, fs_m=70, duration=2**19/70,
            jid='abcdef123', result_dir=None, working_dir='/dev/shm/womo1998/', skip_existing=False):
    
    if result_dir is None:
        result_dir = os.getcwd()
        
    return mapping(zeta, Iy, Iz, alpha, v_b, fs_m, duration, jid, result_dir, working_dir, skip_existing)

def stage1mapping(v_b, jid, result_dir, working_dir):
    
    zeta = 0.002 # normal damping
    # zeta = 0.0005 # light damping
    # zeta = 0.005 # heavy damping
    Iy = 0.01196
    Iz = 0.01304
    alpha = 45

    fs_m = 70
    duration = 2**19/fs_m
    
    skip_existing=True
    
    return mapping(zeta, Iy, Iz, alpha, v_b, fs_m, duration, jid, result_dir, working_dir, skip_existing)

def mapping(zeta, Iy, Iz, alpha, 
            v_b, fs_m, duration,
            jid, result_dir, working_dir, skip_existing):
    print(jid)
    if not isinstance(result_dir, Path):
        result_dir = Path(result_dir)
    
    if not isinstance(working_dir, Path):
        working_dir = Path(working_dir)
    
    # Set up directories
    if '_' in jid:
        id_ale, id_epi = jid.split('_')
        this_result_dir = result_dir / id_ale
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
            
        this_result_dir = this_result_dir / id_epi
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
    else:
        this_result_dir = result_dir / jid
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
    
    if os.path.exists(this_result_dir / 'response.npz') and os.path.exists(this_result_dir / 'excitation.npz') and skip_existing:
        try:
            arr = np.load(this_result_dir / 'response.npz')
            t_vals = arr['t_vals']
            d_freq_time = arr['d_freq_time']
            v_freq_time = arr['v_freq_time']
            a_freq_time = arr['a_freq_time']
        except EOFError as e:
            os.remove(this_result_dir / 'response.npz')
            raise e
        
        arr = np.load(this_result_dir / 'excitation.npz')
        Fu_time = arr['Fu_time']
        Fv_time = arr['Fv_time']
        
        num_nodes = Fu_time.shape[1]
        
    else:
        raise RuntimeError(f'{jid} must be computed')
        seed = int.from_bytes(bytes(jid, 'utf-8'), 'big')
        
        # assemble structural parameters
        struct_parms = {
                'L'         : 200,
                
                'E'         : 2.1e11,
                'A'         : 0.03287,
    
                'x_knl'     : 160,
    
                'm_tmd'     : 800,
                'ky_tmd'    : 1025.48,
                'kz_tmd'    : 1025.48,
                'dy_tmd'    : 200,
                'dz_tmd'    : 200,
                'x_tmd'     : 200,
                }
        
        # Cross-section
        Iyz = 0
        # rotate cross section about angle alpha
        alpha = alpha * 2 * np.pi / 360
        struct_parms['Iy'] = 0.5 * (Iy + Iz) + 0.5 * (Iy - Iz) * np.cos(2 * alpha) + Iyz * np.sin(2 * alpha)
        struct_parms['Iz'] = 0.5 * (Iy + Iz) - 0.5 * (Iy - Iz) * np.cos(2 * alpha) - Iyz * np.sin(2 * alpha)
        struct_parms['Iyz'] = -0.5 * (Iy - Iz) * np.sin(2 * alpha) + Iyz * np.cos(2 * alpha)
        
        # equivalent spring stiffness of guy cables
        A_wire = 0.00075
        rho = 7850
        N_wire = 60000
        Aeq = A_wire/(1 + (A_wire * rho * 9.819 * 70 / N_wire)**2 * struct_parms['E'] * A_wire / 12 / N_wire)
        keq = (struct_parms['E'] * Aeq * (70**2 / (70**2 + 160**2)) + N_wire) / np.sqrt(70**2 + 160**2)
        struct_parms['kz_nl'] = 1.7 * keq
        struct_parms['ky_nl'] = 2 * keq
        
        # additional mass into mass density
        add_mass = 60
        struct_parms['rho'] = rho + add_mass / struct_parms['A']
    
        # model parameters
        num_nodes = 201
        num_modes = 38 # ensure max(f) is below nyquist frequency to avoid wrap-around in the frf
        
        # load or build structural model
        mech = None
        if skip_existing and os.path.exists(result_dir / f'mechanical.npz'):
            try:
                mech = MechanicalDummy.load(fpath=result_dir / f'mechanical.npz')
                assert mech.struct_parms['Iy'] == struct_parms['Iy']
                assert mech.struct_parms['Iz'] == struct_parms['Iz']
                assert mech.struct_parms['Iyz'] == struct_parms['Iyz']
                assert mech.damping == zeta
                assert mech.omegas is not None
                assert mech.frf is not None
            except Exception as e:
                mech = None
                raise e
            
        if mech is None:
            global ansys
            if 'ansys' not in globals():
                ansys = Mechanical.start_ansys(working_dir=working_dir, jid=jid)
            mech = Mechanical(ansys = ansys, jobname=jid, wdir = working_dir)
            mech.build_conti(struct_parms, 
                             Ldiv = num_nodes, damping = zeta, 
                             num_modes = num_modes)
        
        # build or load windfield
        if os.path.exists(this_result_dir / 'excitation.npz') and skip_existing:
            arr = np.load(this_result_dir / 'excitation.npz')
            Fu_time = arr['Fu_time']
            Fv_time = arr['Fv_time']
        else:
            x_grid = mech.nodes_coordinates[0:-2,1]
            
            category = 3
            C_uz = 10
            C_vz = 7
            b = 1.9
            cscd = 1.0
            cf = 2.86519

            Fu_time, Fv_time = windfield(x_grid, 
                                         category, v_b, 
                                         fs_m, duration, 
                                         C_uz, C_vz, 
                                         b, cscd, cf,
                                         seed)
            
            np.savez(this_result_dir / 'excitation.npz',
                     Fu_time=Fu_time, Fv_time=Fv_time)
        
        if True:
            # compute response
            t_vals, response = mech.transient_ifrf(Fu_time, Fv_time,
                                                   mech.nodes_coordinates[0:-2,0],
                                                   inp_dt = 1 / fs_m)
            
            [d_freq_time, v_freq_time, a_freq_time] = response
            
            np.savez(this_result_dir / 'response.npz', 
                     t_vals = t_vals, 
                     d_freq_time = d_freq_time, 
                     v_freq_time = v_freq_time, 
                     a_freq_time = a_freq_time)
            
            # FRF was not pre-computed: save, clear and exit
            if isinstance(mech, Mechanical):
                mech.save(result_dir / f'mechanical.npz')
                ansys.finish()
                ansys.cwd('/dev/shm/womo1998/')
                ansys.clear()
    
    Force_magnitude = np.sqrt(Fu_time**2 + Fv_time**2)
    mean_Force_magnitude = np.mean(Force_magnitude, axis=0)
    std_Force_magnitude = np.std(Force_magnitude, axis=0)
    
    Force_direction = np.arctan2(Fv_time, Fu_time)
    
    mean_Force_direction = np.empty((num_nodes,))
    for node in range(num_nodes):
        vector = np.array([Fu_time[:,node], Fv_time[:,node]]).T       
        _, _, V_T = np.linalg.svd(vector, full_matrices=False)
        mean_Force_direction[node] = np.arctan2(-V_T[1,0], V_T[1,1])
    Force_direction -= mean_Force_direction[np.newaxis, :]
    Force_direction[Force_direction<-np.pi/2] += np.pi
    Force_direction[Force_direction> np.pi/2] -= np.pi
    std_Force_direction = np.std(Force_direction, axis=0)

    RMS_d = np.sqrt(np.mean(d_freq_time**2, axis=0))
    RMS_v = np.sqrt(np.mean(v_freq_time**2, axis=0))
    RMS_a = np.sqrt(np.mean(a_freq_time**2, axis=0))
    
    
    return mean_Force_magnitude, std_Force_magnitude, mean_Force_direction, std_Force_direction, RMS_d, RMS_v, RMS_a


def stage1bmapping(mech, jid, result_dir, skip_existing) :
    
    if not isinstance(result_dir, Path):
        result_dir = Path(result_dir)
    
    # Set up directories
    if '_' in jid:
        id_ale, id_epi = jid.split('_')
        this_result_dir = result_dir / id_ale
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
            
        this_result_dir = this_result_dir / id_epi
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
    else:
        this_result_dir = result_dir / jid
        if not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
    
    if os.path.exists(this_result_dir / 'response.npz') and skip_existing:
        return
        arr = np.load(this_result_dir / 'response.npz')
        t_vals = arr['t_vals']
        d_freq_time = arr['d_freq_time']
        v_freq_time = arr['v_freq_time']
        a_freq_time = arr['a_freq_time']
        return
    
    elif os.path.exists(this_result_dir / 'excitation.npz'):
        arr = np.load(this_result_dir / 'excitation.npz')
        Fu_time = arr['Fu_time']
        Fv_time = arr['Fv_time']
        
        
        # compute response
        now = time.time()
        t_vals, response = mech.transient_ifrf(Fu_time, Fv_time,
                                               mech.nodes_coordinates[0:-2,0],
                                               inp_dt = 1 / 70)
        
        print(f'Computed response on {jid} in {time.time()-now:1.2f} s')
        
        [d_freq_time, v_freq_time, a_freq_time] = response
        
        np.savez(this_result_dir / 'response.npz', 
                 t_vals = t_vals, 
                 d_freq_time = d_freq_time, 
                 v_freq_time = v_freq_time, 
                 a_freq_time = a_freq_time)
    else:
        print(f'Excitation not present for {jid}')
        
        return
        
            
    return 

def default_windfield(x_grid):
    return windfield(x_grid, 
              category=3, v_b=25.0, 
              fs_w=70, duration=2**19/70, 
              C_uz=10, C_vz=7, 
              b=1.9, cscd=1.0, cf=2.86519,
              seed=None)

def windfield(x_grid, category, v_b, fs_w, duration, C_uz, C_vz, b, cscd, cf, seed=None):
    # wind would not have to be generated up to higher frequencies, 
    # as the energy in these bands is negligible
    # but that also prevents any model response in this range with FRF based methods
    # response sample rate can be adjusted by zero padding the fft

    N = int(duration * fs_w)
    if not (N != 0) and (N & (N-1) == 0):
        logger.warning(f'The number of timesteps N={N} is not a power of two. FFT performance will degrade.')

    # Frequency domain grid
    f_w = np.fft.rfftfreq(N, 1/fs_w)[:,np.newaxis]
    
    # Spatial domain grid

    # Geländekategorie I - IV
    z_min, alpha, vm_fact, vm_min, Iv_fact, Iv_min, eps = terrain_parameters(category)

    # Windzone 1 - 4
    # zone = 2
    # v_b = [None, 22.5, 25.0, 27.5, 30][zone] # m/s Basiswindgeschwindigkeit (v_b = v_b0 in DE vgl. NA S. 5)

    v_m, sigma_v, L = basic_wind_parameters(x_grid, v_b, z_min, alpha, vm_fact, vm_min, Iv_fact, Iv_min, eps)

    u_freq, v_freq = spectral_wind_field(x_grid, f_w, 
                                         L, v_m, sigma_v, C_uz, C_vz,
                                         seed)

    u_time, v_time = temporal_wind_field(u_freq, v_freq, N)
    
    # including mean wind
    # F_uj, F_vj = force_wind_field(u_j + v_m[:,np.newaxis], v_j, delta_x=x_grid[1]-x_grid[0], 
    #                               b=1.9, cscd=1.0, cf=2.86519, rho=1.25)
    # not including mean wind
    Fu_time, Fv_time = force_wind_field(u_time, v_time, x_grid[1]-x_grid[0], b, cscd, cf)
    
    return Fu_time, Fv_time


def plot_response_field(d_freq_time=None, v_freq_time=None, a_freq_time=None, duration=None, height=None):

    for arr in [d_freq_time, v_freq_time, a_freq_time]:
        if arr is not None:
            if duration is None:
                duration = arr.shape[1]
            if height is None:
                height = arr.shape[0]
            break
            
    extent = (0, duration, 0, height)
    
    if d_freq_time is not None:
        fig1, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        vmin, vmax = np.min(d_freq_time), np.max(d_freq_time)
        im1 = axes[0].imshow(d_freq_time[:,:,0].T, origin='lower', aspect='auto', extent=(0, duration, 0, height), vmin=vmin, vmax=vmax, label='y')
        im2 = axes[1].imshow(d_freq_time[:,:,1].T, origin='lower', aspect='auto', extent=(0, duration, 0, height), vmin=vmin, vmax=vmax, label='z')
        fig1.colorbar(im2, ax=axes).set_label('Displacement [m]')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Height [m]')
        axes[1].legend(title='$d_z$')
        axes[0].set_ylabel('Height [m]')
        axes[0].legend(title='$d_y$')
    else:
        fig1 = None

    if v_freq_time is not None:
        fig2, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        vmin, vmax = np.min(v_freq_time), np.max(v_freq_time)
        im1 = axes[0].imshow(v_freq_time[:,:,0].T, origin='lower', aspect='auto', extent=(0, duration, 0, height), vmin=vmin, vmax=vmax, label='y')
        im2 = axes[1].imshow(v_freq_time[:,:,1].T, origin='lower', aspect='auto', extent=(0, duration, 0, height), vmin=vmin, vmax=vmax, label='z')
        fig2.colorbar(im2, ax=axes).set_label('Velocity [m/s]')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Height [m]')
        axes[1].legend(title='$v_z$')
        axes[0].set_ylabel('Height [m]')
        axes[0].legend(title='$v_y$')
    else:
        fig2 = None
    
    if a_freq_time is not None:
        fig3, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        vmin, vmax = np.min(a_freq_time), np.max(a_freq_time)
        im1 = axes[0].imshow(a_freq_time[:,:,0].T, origin='lower', aspect='auto', extent=(0, duration, 0, height), vmin=vmin, vmax=vmax, label='y')
        im2 = axes[1].imshow(a_freq_time[:,:,1].T, origin='lower', aspect='auto', extent=(0, duration, 0, height), vmin=vmin, vmax=vmax, label='z')
        fig3.colorbar(im2, ax=axes).set_label('Acceleration [m/s^2]')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Height [m]')
        axes[1].legend(title='$a_z$')
        axes[0].set_ylabel('Height [m]')
        axes[0].legend(title='$a_y$')
    else:
        fig3 = None
    
    return fig1, fig2, fig3

def animate_response(nodes_coordinates, d_time=None, d_freq_time=None, Fu_time=None, Fv_time=None, N_frames=None):

    # compare time histories by animating response side-by-side
    from matplotlib.animation import FuncAnimation
    
    x_grid = nodes_coordinates[0:-2,1]
    num_nodes = nodes_coordinates.shape[0] - 2
    ind = np.ones(num_nodes + 2, dtype=bool)
    ind[-2] = False

    if d_time is not None: # transient results
        uy1 = d_time[:,ind,0]
        uz1 = d_time[:,ind,1]
    else:
        uy1,uz1=None,None
        
    if d_freq_time is not None: # FRF results
        uy2 = d_freq_time[:,ind,0]
        uz2 = d_freq_time[:,ind,1]
    else:
        uy2,uz2=None,None    

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    z = x_grid
    zero = np.zeros_like(z)

    if uy1 is not None:
        lines_response1 = ax.plot(uy1[0, :], uz1[0, :], nodes_coordinates[ind,1], alpha=0.6)[0]
        lim = np.max([-np.min(uy1), np.max(uy1), -np.min(uz1), np.max(uz1)])*20
    else:
        lines_response1 = None

    if uy2 is not None:
        lines_response2 = ax.plot(uy2[0, :], uz2[0, :], nodes_coordinates[ind,1], alpha=0.6)[0]
        lim = np.max([-np.min(uy2), np.max(uy2), -np.min(uz2), np.max(uz2)])*20
    else:
        lines_response2 = None

    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))

    if Fu_time is not None and Fv_time is not None:
        Flim = np.max([-np.min(Fu_time), np.max(Fu_time), -np.min(Fv_time), np.max(Fv_time)])

        F_uj_inter = np.empty((3*len(x_grid),), dtype=Fu_time.dtype)
        F_uj_inter[0::3] = zero
        F_uj_inter[1::3] = Fu_time[0, :]*lim/Flim
        F_uj_inter[2::3] = zero

        F_vj_inter = np.empty((3*len(x_grid),), dtype=Fv_time.dtype)
        F_vj_inter[0::3] = zero
        F_vj_inter[1::3] = Fv_time[0, :]*lim/Flim
        F_vj_inter[2::3] = zero

        x_grid_inter = np.empty((3*len(x_grid),), dtype=x_grid.dtype)
        x_grid_inter[0::3] = x_grid
        x_grid_inter[1::3] = x_grid
        x_grid_inter[2::3] = x_grid

        lines_force = ax.plot(F_uj_inter, F_vj_inter, x_grid_inter, alpha=0.3)[0]
    else:
        lines_force = None

    # lines_force+lines_response

    def update(n):
        if lines_force is not None:
            F_uj_inter[1::3] = Fu_time[n, :]*lim/Flim
            F_vj_inter[1::3] = Fv_time[n, :]*lim/Flim
            lines_force.set_data(F_uj_inter, F_vj_inter)
            lines_force.set_3d_properties(x_grid_inter)
        if uy1 is not None:
            lines_response1.set_data(uy1[n, :], uz1[n, :])
            lines_response1.set_3d_properties(nodes_coordinates[ind,1])
        if uy2 is not None:
            lines_response2.set_data(uy2[n, :], uz2[n, :])
            lines_response2.set_3d_properties(nodes_coordinates[ind,1])

        return lines_force

    ani = FuncAnimation(fig, update, frames=N_frames, blit=True, interval=33, repeat=False)
    return fig, ani



def vars_definition(stage=2):
    
    lamda = MassFunction('lambda_vb',[(2.267, 2.3),(1.96, 2.01)],[0.75,0.25], primary=False) # incompleteness
    c = MassFunction('c_vb',[(5.618, 5.649),(5.91,6.0)],[0.75,0.25], primary=False) # incompleteness
    
    v_b = RandomVariable('weibull_min','v_b', [lamda, c], primary=True) # meter per second
    alpha = RandomVariable('uniform', 'alpha', [0., 180.], primary=True) # degreee
    
    n_locations = MassFunction('n_locations', [(4,), (8,), (12,)], [0.2, 0.5, 0.3], primary=True)
    
    DTC = MassFunction('DTC', [(1.6, 30), (0.8, 1.6), (0.4, 0.8)],[0.7, 0.2, 0.1], primary=True)
    
    #= MassFunction('', [(,), ], [], primary=True)
    
    sensitivity_nominal = MassFunction('sensitivity_nominal', [(1.02,), (0.102, 1.02), (0.01, 0.51)], [0.5, 0.3, 0.2], primary=False)
    # ..TODO:: actually, the last mass should be 0.1, but that requires several modifications to include the open-world assumption 
    sensitivity_deviation = MassFunction('sensitivity_deviation', [(5.,), (10.,), (2., 20.)], [0.4, 0.4, 0.2, ], primary=False)
    # Formal definition of sensitivity would result in a sample of n_channels x N_mcs_epi, where n_channels changes for each n_epi
    # Actual implementation is in Acquire.apply_sensor, using a seed, derived from the sample ID
    #sensitivity = RandomVariable('Uniform', 'sensitivity', 
    #                             [sensitivity_nominal - sensitivity_deviation, sensitivity_nominal + sensitivity_deviation, n_channels], primary =True)
    # make MassFunctions primary, to pass them on to the mapping
    sensitivity_nominal.primary = True
    sensitivity_deviation.primary = True
    
    spectral_noise_slope = MassFunction('spectral_noise_slope', [(-0.8,-0.3),], [1.0,], primary=True)
    sensor_noise_rms = MassFunction('sensor_noise_rms', [(1e-6,1e-2,),], [1.0,], primary=True)
    # Formal definition of spectral_noise would result in a sample of n_channels x num_timesteps x N_mcs_epi, where n_channels, num_timesteps changes for each n_epi
    # Actual implementation is in Acquire.apply_sensor, using a seed, derived from the sample ID
    #spectral_noise = RandomVariable('normal', 'spectral_noise', [...], primary=True)
    # make MassFunctions primary, to pass them on to the mapping
    spectral_noise_slope.primary = True
    sensor_noise_rms.primary = True
    
    range_estimation_duration = MassFunction('range_estimation_duration', [(30.,), (60, 120), (300,)], [0.2, 0.5, 0.3], primary=False)
    range_estimation_margin = MassFunction('range_estimation_margin', [(2., 5.), (5., 10.)], [0.6, 0.4], primary=False)
    # Formal definition of meas_range is based on the signal itself and cannot be done using simple RandomVariabls
    # Actual implementation is in Acquire.estimate_meas_range, using a seed, derived from the sample ID
    #meas_range = RandomVariable(..., primary=True)
    # make MassFunctions primary, to pass them on to the mapping
    range_estimation_duration.primary = True
    range_estimation_margin.primary = True
    
    DAQ_noise_rms = MassFunction('DAQ_noise_rms', [(2.5e-6, 3e-3),], [1.0, ], primary=False)
    # Formal definition of DAQ_noise would result in a sample of n_channels x num_timesteps x N_mcs_epi, where n_channels, num_timesteps changes for each n_epi
    # Actual implementation is in Acquire.add_noise, using a seed, derived from the sample ID
    #DAQ_noise = RandomVariable('DAQ_noise', 'normal', [0, DAQ_noise_rms, (n_channels, num_timesteps)], primary=True)
    # make MassFunctions primary, to pass them on to the mapping
    DAQ_noise_rms.primary = True
    
    # sampling_rate = MassFunction('sampling_rate', [(50.,100.), (10.,50.), (4.,10.)], [0.5, 0.3, 0.2], primary=True)
    # logger.warning('The decimation factors are too high for a final estimation up to 3.5 Hz (numerical pre-study). Consider modifications')
    decimation_factor = MassFunction('decimation_factor', [(1, 2), (2,7), (7, 18)], [0.5, 0.3, 0.2], primary=True)

    anti_aliasing_cutoff_factor = MassFunction('anti_aliasing_cutoff_factor', [(0.4, 0.45), (0.45, 0.49), (0.5,)], [0.7, 0.2, 0.1], primary=True)
    
    
    quant_bit_factor = MassFunction('quant_bit_factor', [(3,), (4,), (6,)], [0.1, 0.3, 0.6], primary=True)
    
    duration = MassFunction('duration', [(10.*60., 20.*60.), (30.*60., 45.*60.), (60.*60.,), (120.*60.,)], [0.1, 0.2, 0.5, 0.2], primary=True)
    
    tau_max = MassFunction('tau_max', [(20.0,175.0),(60.0,175.0)], [0.5, 0.5], primary=True)
    m_lags = MassFunction('m_lags', [(20,1000),(50,300)], [0.4, 0.6], primary=True)    
    
    model_order = MassFunction('model_order', [(10,100),], [1.0,], primary=True)
    
    estimator = MassFunction('estimator', [(0, 1),], [1.0, ]) # 0: welch, 1: blackman-tukey
    
    
    if stage==1:
        vars_epi = [lamda, c]
        vars_ale = [v_b, alpha] 
        
        arg_vars = {'v_b':v_b.name,} # 9 
        
    elif stage==2:
        vars_epi = [lamda, c, # stage 1
                    n_locations, DTC, sensitivity_nominal, sensitivity_deviation, 
                    spectral_noise_slope, sensor_noise_rms,
                    range_estimation_duration, range_estimation_margin,
                    DAQ_noise_rms,
                    decimation_factor, anti_aliasing_cutoff_factor,
                    quant_bit_factor, duration]
        vars_ale = [v_b, alpha] # all stage 1
        
        arg_vars = {'n_locations':n_locations.name, 
                    'DTC':DTC.name,  
                    'sensitivity_nominal':sensitivity_nominal.name,  
                    'sensitivity_deviation_percent':sensitivity_deviation.name,  
                    'spectral_noise_slope':spectral_noise_slope.name,  
                    'sensor_noise_rms':sensor_noise_rms.name, 
                    'range_estimation_duration':range_estimation_duration.name,  
                    'range_estimation_margin':range_estimation_margin.name, 
                    'DAQ_noise_rms':DAQ_noise_rms.name, 
                    'decimation_factor':decimation_factor.name,  
                    'anti_aliasing_cutoff_factor':anti_aliasing_cutoff_factor.name, 
                    'quant_bit_factor':quant_bit_factor.name, 
                    'duration':duration.name,}
        
    elif stage==3:
        vars_epi = [lamda, c, # stage 1
                    n_locations, DTC, sensitivity_nominal, sensitivity_deviation, 
                    spectral_noise_slope, sensor_noise_rms,
                    range_estimation_duration, range_estimation_margin,
                    DAQ_noise_rms,
                    decimation_factor, anti_aliasing_cutoff_factor,
                    quant_bit_factor, duration, #stage 2
                    m_lags, estimator, model_order, tau_max] # stage 3
        vars_ale = [v_b, alpha] # all stage 1
        
        arg_vars = {'n_locations':n_locations.name, 
                    'DTC':DTC.name,  
                    'sensitivity_nominal':sensitivity_nominal.name,  
                    'sensitivity_deviation_percent':sensitivity_deviation.name,  
                    'spectral_noise_slope':spectral_noise_slope.name,  
                    'sensor_noise_rms':sensor_noise_rms.name, 
                    'range_estimation_duration':range_estimation_duration.name,  
                    'range_estimation_margin':range_estimation_margin.name, 
                    'DAQ_noise_rms':DAQ_noise_rms.name, 
                    'decimation_factor':decimation_factor.name,  
                    'anti_aliasing_cutoff_factor':anti_aliasing_cutoff_factor.name, 
                    'quant_bit_factor':quant_bit_factor.name, 
                    'duration':duration.name,
                    'm_lags':m_lags.name,
                    'estimator':estimator.name,
                    'model_order':model_order.name,
                    }
    
    return vars_ale, vars_epi, arg_vars


def main():
    import glob
    
    result_dir = Path('/usr/scratch4/sima9999/work/modal_uq/uq_oma_a/samples')
    working_dir = Path('/dev/shm/womo1998/')
    
    mech = MechanicalDummy.load(fpath=result_dir / f'mechanical.npz')

    
    flist = glob.glob('/usr/scratch4/sima9999/work/modal_uq/uq_oma_a/samples/*/*')
    flist = glob.glob('/usr/scratch4/sima9999/work/modal_uq/uq_oma_a/samples/*/*')
    todolist = []
    for file in flist:
        if not os.path.exists(file+'/response.npz'):
            todolist.append(file)
    
    chunksize = int(len(todolist)/10)
    this_chunk = int(sys.argv[1])
    
    for path in todolist[chunksize*this_chunk:chunksize*(this_chunk+1)]:
        path, eid = os.path.split(path)
        path, aid = os.path.split(path)
        if eid != 'e3f6077f': 
            continue
        jid = aid + '_' + eid
        try:
            stage1bmapping(mech, jid, result_dir, True)
        except Exception as e:
            print(e)
        
    
    
    # default_mapping(jid='8a2a343d_e3f6077f',
    #                 result_dir=Path('/usr/scratch4/sima9999/work/modal_uq/uq_oma_a/samples/'),
    #                 skip_existing=True)
    print('exit')
    
if __name__ == '__main__':
    vars_definition(stage=2)