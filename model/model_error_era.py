import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import logging
from model.mechanical import generate_mdof_time_hist, start_ansys
from model.acquisition import Acquire
from pyOMA.core.SSICovRef import BRSSICovRef
from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor

matplotlib.use('qt5Agg')
logging.basicConfig(level=logging.INFO)
# Modal Analysis PostProcessing Class e.g. Stabilization Diagram
#from StabilDiagram import StabilCalc, StabilPlot, StabilGUI, start_stabil_gui

# Modeshape Plot
#from PlotMSH import ModeShapePlot, start_msh_gui


print_context_dict = {'text.usetex': True,
                      'text.latex.preamble': "\\usepackage{siunitx}\n \\usepackage{xfrac}",
                      'font.size': 10,
                      'legend.fontsize': 10,
                      'xtick.labelsize': 10,
                      'ytick.labelsize': 10,
                      'axes.labelsize': 10,
                      'font.family': 'serif',
                      'legend.labelspacing': 0.1,
                      'axes.linewidth': 0.5,
                      'xtick.major.width': 0.2,
                      'ytick.major.width': 0.2,
                      'xtick.major.width': 0.5,
                      'ytick.major.width': 0.5,
                      'figure.figsize': (5.906, 5.906 / 1.618),
                      'figure.dpi': 100}

# must be manually set due to some matplotlib bugs
if print_context_dict['text.usetex']:
    # plt.rc('text.latex',unicode=True)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble="\\usepackage{siunitx}\n \\usepackage{xfrac}")


def model_error_ERA_SSI(jid, result_dir, working_dir,
                        num_nodes, num_modes, damping, freq_scale,
                        imp_scale,
                        dt_fact, num_cycles, num_ref_nodes, num_meas_nodes,
                        nl_stiff=None, out_quant=['d'],
                        snr_db=0, dec_fact=1,
                        model_order=None,
                        skip_existing=True,
                        **kwargs):
    '''
    A function to evaluate the model error from numerical structural 
    dynamics over acquisition and signal processing to 
    deterministic/stochastic signal processing
    
    Intended to be run multiple times in:
        full factorial framework
        elementary effects framework
        a stochastic (quasi) Monte Carlo framework 
        a polymorphic uncertainty framework (Monte Carlo nested with interval optimization or Monte Carlo)
    
    for uncertainty quantifications and sensitiviy analyses
    
    The computational flow and required input parameters are as follows:
    
    1. IRF-matrix [accel or disp, stepped impulse -> same as free decay]: 
            parameters (num_nodes, 
                        damping # global rayleigh, later friction/hysteretic 
                        freq_scale, 
                        num_modes, # ideally computed from a ratio of num_nodes beforehand
                        imp_scale, # same for each ref_node, not relevant in linear analyses
                        dt_fact, # [1e-1...1e1] for fmax 
                        num_cycles, # [3...20] for fmin 
                        num_ref_nodes, # evenly distributed from top
                        num_meas_nodes, # evenly distributed from top
                        [nl_stiff],
                        [fric/hyst damp],
                        (out_quant),
                        )
    
    2. Acquisition/Processing [offset_removal, filtering, (windowing), noise, decimation]:
            parameters ([filter cut-off mode == num_modes],
                        [type of window function],
                        snr_db (per channel, global), 
                        decimation factor, 
                        ) 
        
    3. System Identification / Modal Analysis [ERA, SSI] 
            parameters ([n_lags==timesteps], 
                         model order # ideally computed from a ratio of num_modes beforehand >=2*num_modes
                        )
    
    The output parameters are:
    
    m: number of modes
    n: number of nodes
    r: number of reference nodes (excitation nodes) (subset of n)
    c: number of channels / number of measurement nodes (subset of n)
    s: scalar values
    
    coordinates: jid, m, n
    
    return  m     compensated modal frequencies
            m     compensated modal damping
            m x n compensated mode shapes
            n     k values
            n     eps_duff values
            n     f_sl values 
            n     hysteresis values (stiffness hysteresis: KEYOPT(0)=1 for combin39)
            m x r impulse works
            c     signal powers
            c     noise powers 
            m     identified modal frequencies
            m     identified modal damping
            m x n identified mode shapes
            m     modal contributions (averaged over n)
            s     deltat
            s     timesteps
            
    
    postprocessing outside of this function might include: 
        frequency differences, 
        damping differences, 
        MAC values (mean and confidence intervals)
    
    next step: ambient-> SSI
    
    
    CONTINUE HERE: Implement acqusition, test/verify
    '''
    ref_nodes = np.rint(np.linspace(2, num_nodes, int(num_ref_nodes))).astype(int)
     # start at 2 since 1 usually is the constrained node
    
    if not os.path.exists(os.path.join(result_dir, 'mechanical.npz')) or not skip_existing:
        ansys = start_ansys(working_dir, jid)
    
        mech = generate_mdof_time_hist(ansys, num_nodes=num_nodes, num_modes=num_modes,
                                       damping=damping, freq_scale=freq_scale,
                                       num_meas_nodes=num_meas_nodes,
                                       nl_stiff=nl_stiff,
                                       just_build=True)
        
        dt_fact, deltat, num_cycles, timesteps, _ = mech.signal_parameters(dt_fact=dt_fact, num_cycles=num_cycles)
    
        
        imp_forces = np.ones((num_ref_nodes,)) * imp_scale
        imp_times = np.ones((num_ref_nodes,)) * deltat
        imp_durs = np.ones((num_ref_nodes,)) * timesteps * deltat
        
        t_vals, IRF_matrix, F_matrix, ener_mat, amp_mat = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form='step', mode='matrix', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles, out_quant=out_quant, **kwargs)
        n_lags = timesteps
        frequencies_comp, damping_comp, modeshapes_comp = mech.numerical_response_parameters()
        meas_nodes = mech.meas_nodes
        
        savedict = {'dt_fact': dt_fact,
                    'deltat': deltat,
                    'num_cycles': num_cycles,
                    'timesteps': timesteps,
                    't_vals': t_vals,
                    'IRF_matrix': IRF_matrix,
                    'F_matrix': F_matrix,
                    'ener_mat': ener_mat,
                    'amp_mat': amp_mat,
                    'n_lags': n_lags,
                    'frequencies_comp': frequencies_comp,
                    'damping_comp': damping_comp,
                    'modeshapes_comp': modeshapes_comp,
                    'meas_nodes': meas_nodes}
        np.savez(os.path.join(result_dir, 'mechanical.npz'), **savedict)
    else:
        arr = np.load(os.path.join(result_dir, 'mechanical.npz'))
        dt_fact = arr['dt_fact']
        deltat = arr['deltat']
        num_cycles = arr['num_cycles']
        timesteps = arr['timesteps']
        t_vals = arr['t_vals']
        IRF_matrix = arr['IRF_matrix']
        F_matrix = arr['F_matrix']
        ener_mat = arr['ener_mat']
        amp_mat = arr['amp_mat']
        n_lags = arr['n_lags']
        frequencies_comp = arr['frequencies_comp']
        damping_comp = arr['damping_comp']
        modeshapes_comp = arr['modeshapes_comp']
        meas_nodes = arr['meas_nodes']
    
    # for rnode_ind, rnod in enumerate(ref_nodes):
        # plt.figure()
        # for mnode_ind, mnode in enumerate(meas_nodes):
            # if not mnode_ind: continue
            #
            # plt.plot(t_vals,IRF_matrix[mnode_ind,rnode_ind,:], label=f'{rnod}-{mnode}')
    # plt.legend()
    # plt.show()
    
    acquisition = Acquire(t_vals, IRF_matrix=IRF_matrix)
    power_signal, power_noise = acquisition.add_noise(snr_db)
    dt,_ = acquisition.sample(f_max=frequencies_comp[-1], fs_factor=2.5)
    t_vals,IRF_matrix = acquisition.get_signal()
    
    

    mech.export_geometry(result_dir)
    geometry_data = GeometryProcessor.load_geometry(f'{result_dir}/grid.txt', f'{result_dir}/lines.txt')
    
    velo_channels = None
    disp_channels = None
    accel_channels = None
    if out_quant[0] == 'd': disp_channels = list(range(num_meas_nodes))
    elif out_quant[0] == 'a': accel_channels = list(range(num_meas_nodes))
    elif out_quant[0] == 'v': velo_channels = list(range(num_meas_nodes))
    
    channel_headers = meas_nodes
    ref_channels = [i for i, node in enumerate(meas_nodes) if node in ref_nodes]
    dummy_meas = np.zeros((timesteps, num_meas_nodes))

    prep_data = PreProcessSignals(dummy_meas, 1 / deltat, ref_channels=ref_channels,
                                                  accel_channels=accel_channels,
                                                  velo_channels=velo_channels,
                                                  disp_channels=disp_channels,
                                                  channel_headers=channel_headers)
    chan_dofs = prep_data.load_chan_dofs(f'{result_dir}/chan_dofs.txt')
    prep_data.add_chan_dofs(chan_dofs)
    prep_data.n_lags = n_lags
    prep_data.corr_matrix = IRF_matrix
    prep_data.corr_matrices = [IRF_matrix]
    prep_data.save_state(f'{result_dir}/prep_data.npz')
    
    modal_data = BRSSICovRef(prep_data)
    num_block_columns=n_lags//2
    modal_data.build_toeplitz_cov(num_block_columns)
    modal_data.compute_state_matrices(model_order)

    frequencies_id, damping_id, modeshapes_id, _, modal_contributions = modal_data.single_order_modal(model_order)
    num_modes_id = model_order//2+model_order%2# valid results only up to half model order
    sort_inds = np.argsort(frequencies_id[:num_modes_id])
    frequencies_id = frequencies_id[sort_inds]
    damping_id = damping_id[sort_inds]
    modal_contributions = modal_contributions[sort_inds]
    modeshapes_id = modeshapes_id[:, sort_inds]

    #mode shape normalization
#     for mode in range(num_modes_id):
#         modeshapes_id[:, mode] /= modeshapes_id[np.argmax(np.abs(modeshapes_id[:, mode])), mode]
#         modeshapes[:, mode] /= modeshapes[np.argmax(np.abs(modeshapes[:, mode])), mode]
        #modeshapes_id[:, mode] /= modeshapes_id[-1, mode]
        #modeshapes[:, mode] /= modeshapes[-1, mode]

    return frequencies_comp, damping_comp, modeshapes_comp, mech.k_vals, mech.eps_duff_vals, mech.sl_force_vals, mech.hyst_vals,ener_mat, power_signal, power_noise, frequencies_id, damping_id, modeshapes_id, modal_contributions, deltat, timesteps

    plt.figure()
    plt.plot(frequencies_comp, ls='none', marker='+')
    plt.plot(frequencies_id, ls='none', marker='x')
    plt.figure()
    plt.plot(damping * 100, ls='none', marker='+')
    plt.plot(damping_id, ls='none', marker='x')
    plt.figure()
    plt.plot(modeshapes_comp)
    plt.plot(modeshapes_id, ls='dotted')
    plt.figure()
    total_energy = np.sum(ener_mat)
    plt.plot(frequencies_comp, np.sum(ener_mat, axis=0) / total_energy, ls='none', marker='+')
    plt.plot(frequencies_id, modal_contributions, ls='none', marker='x')
    plt.show()

def main():
    import uuid
    
    working_dir = '/dev/shm/womo1998/'
    os.makedirs(working_dir, exist_ok=True)
    jid = str(uuid.uuid4()).split('-')[-1]
    result_dir = '/usr/scratch4/sima9999/work/modal_uq/model_error_era/'
    num_nodes = 10
    num_modes = 3
    damping = 0.01
    freq_scale = 1
    
    num_cycles = 15
    dt_fact = 0.02
    imp_scale = 1e8
    num_ref_nodes = 3
    num_meas_nodes = 8
    nl_stiff=None
    out_quant = ['d']
    snr_db = 20
    dec_fact = 2.5/dt_fact
    model_order = 3*num_nodes
        
    model_error_ERA_SSI(jid, result_dir, working_dir, 
                        num_nodes, num_modes, damping, freq_scale, 
                        imp_scale, 
                        dt_fact, num_cycles, 
                        num_ref_nodes, num_meas_nodes, 
                        nl_stiff, out_quant, 
                        snr_db, dec_fact, 
                        model_order,
                        skip_existing=True)

if __name__ == '__main__':
    main()