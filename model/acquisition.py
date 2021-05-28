'''
models the signal acquisition:




'''

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os


class Acquire(object):
    '''
    A class for processing a signal, simulating ADC signal acquisition:
        apply sensor FRFs
        add measurement noise
        sample (apply lowpass AA filter and downsample)
    
    Additionally, characteristics of the generating system and the signal
    itself are modified according to the respective operations:
        - number of modes -> changes due to downsampling
        - modal frequencies -> reduces to new number of modes
        - modal damping -> reduces to new number of modes, might change due to windowing?
        - mode shapes -> reduces to new number of modes, should not be affected, because operators are applied equally to all channels
        - modal amplitudes -> reduces to new number of modes, changes according to sensor frf and SNR
        - signal-to-noise ratio -> changes according to the amount of noise added
        - decimation noise -> results from imperfect filtering before sample reduction
        - signal energies -> might change due to frf, noise and sampling
        - ...
        
    '''
    def __init__(self, t_vals, signal=None, IRF_matrix=None, 
                 modal_frequencies=None, modal_damping=None, mode_shapes=None
                 ):
        '''
        input a synthethised signal (ambient response, impulse response) and corresponding t_vals 
        
        
        
        '''
        if IRF_matrix is not None:
            assert signal is None
            num_meas_nodes = IRF_matrix.shape[0]
            num_ref_nodes = IRF_matrix.shape[1]
            num_timesteps = IRF_matrix.shape[2]
            signal = IRF_matrix.reshape((num_meas_nodes * num_ref_nodes, num_timesteps))
            self.re_shape = (num_meas_nodes, num_ref_nodes, num_timesteps)
        else:
            self.re_shape = signal.shape
            
        self.num_channels = signal.shape[0]
        self.num_timesteps = signal.shape[1]
        
        self.deltat = np.mean(np.diff(t_vals))
        
        assert self.num_timesteps == len(t_vals)
        
        self.t_vals = t_vals
        self.signal = signal
        
        if modal_frequencies is not None:
            assert isinstance(modal_frequencies, (list, tuple, np.ndarray))
            assert len(modal_frequencies.shape) == 1
            
            self.modal_frequencies = modal_frequencies
            self.num_modes = modal_frequencies.size
        
        if modal_damping is not None:
            assert isinstance(modal_damping, (list, tuple, np.ndarray))
            assert len(modal_damping.shape) == 1
            
            self.modal_damping = modal_damping
            self.num_modes = modal_damping.size
            
        if mode_shapes is not None:
            assert isinstance(mode_shapes, (list, tuple, np.ndarray))
            assert len(mode_shapes.shape) == 2 # needs reduced mode shapes (channel,mode) type, not (node, dof, mode) 3D type
            
            self.mode_shapes = mode_shapes
            self.num_modes = mode_shapes.shape[1]
            assert mode_shapes.shape[0] == self.num_channels
            
        return cls
    
    @classmethod
    def init_from_mech(cls, mech):
        
        '''
        Extract the relevant parameters from a mechanical object
        and keeps a reference to that object?
        
        That assumes, only one type of signal (free decay, ambient, impulse)
        is present in the mechanical object (uses the first encountered type)
        
        '''
        
        if mech.state[5]:
            modal_frequencies = mech.frequencies_comp
            modal_damping = mech.modal_damping_comp
            mode_shapes = mech.mode_shapes_comp
        elif mech.state[4]:
            modal_frequencies = mech.damped_frequencies
            modal_damping = mech.modal_damping
            mode_shapes = mech.damped_mode_shapes
        else:
            modal_frequencies = None
            modal_damping = None
            mode_shapes = None
        
        if mech.state[1]:
            t_vals = mech.t_vals_decay
            signal = mech.resp_hist_decay
            acqui = cls(t_vals, signal, None, modal_frequencies, modal_damping, mode_shapes)
            
        if mech.state[2]:
            #mech.inp_hist_amb
            t_vals = mech.t_vals_amb
            signal = mech.resp_hist_amb
            acqui = cls(t_vals, signal, None, modal_frequencies, modal_damping, mode_shapes)

        if mech.state[3]:
            #mech.inp_hist_imp
            t_vals = mech.t_vals_imp
            signal = mech.resp_hist_imp
            modal_energies = mech.modal_imp_energies
            modal_amplitudes = mech.modal_imp_amplitudes
            acqui = cls(t_vals, signal, None, modal_frequencies, modal_damping, mode_shapes)
            
        if mech.state[6]:
            t_vals = mech.t_vals_imp
            IRF_matrix = mech.IRF_matrix
            #mech.imp_hist_imp_matrix
            modal_energies = mech.modal_imp_energy_matrix
            modal_amplitudes = mech.modal_imp_amplitude_matrix
            acqui = cls(t_vals, None, IRF_matrix, modal_frequencies, modal_damping, mode_shapes)
        
        #mech.trans_params
        assert acqui.deltat == mech.deltat
        assert acqui.num_timesteps == mech.timesteps
                
        if mech.state[4] or mech.state[5]:
            assert acqui.num_modes == mech.num_modes
        
        return acqui
            
    def apply_FRF(self, type=None):
        '''
        (- apply sensor FRF (get frfs from our real sensors: 10V/g, 5V/g, small ones, geophones, laser-vib, won't use displacement) by discrete convolution in time domain)
        '''
        pass
    
    def sample(self, f_max, fs_factor, ):
        '''
        apply an (simulated) analog anti-aliasing filter
            use a hardcoded FIR brickwall filter, to avoid any undesired effects from aliasing,
            as this is not the primary objective of the research
        then sample-and-hold -> take every nth sample
        
        
        - define highest mode frequency to be considered (there may be higher modes present in the signal) f_max
        - define oversampling factor for highest mode (f_s = fs_factor*f_max) (fs_factor = [2 ... 20?]
        - apply an (simulated analog) anti-aliasing filter with 3dB passband at fs_/2  
        - define filter type, filter order, (filter 3dB passband)
        https://www.digikey.de/en/articles/the-basics-of-anti-aliasing-low-pass-filters
        - sample and hold / zero-order hold (decimation to sample rate) -> take every nth value

        '''
        
        scipy.signal.fir
        #slicing tuple -> start:stop:step = 0:samples_ful_blocks:block_length
        self.signal = self.signal[0:N_dec*dec_fact:dec_fact, :]
        dt = (self.signal[-1, 0] - self.signal[0, 0]) / (self.signal.shape[0] - 1)
        
        self.re_shape[-1] = self.signal.shape[-1]
        
        return dt
    
    def add_noise(self, snr_db=np.infty, noise_power=None):
        '''
        Add noise to a signal either before or after sampling
        snr_db or noise power can be global or per channel
        if snr_db is given it overrides noise_power
        
        returns power_signal, power_noise
        
        '''
        snr = 10 ** (snr_db / 10)
    
        power = np.mean(array[:, 1] ** 2)
    
        # decimate
        N = array.shape[0]
        # add noise
        noise_power = power / snr
        noise = np.random.normal(0, np.sqrt(noise_power), N)
        power_noise = np.mean(noise ** 2)
    
        snr_actual = power / power_noise
        snr_actual_db = 10 * np.log10(snr_actual)
    
        array[:, 1] += noise

        return power, power_noise
    
    def get_signal(self):
        
        return self.t_vals, self.signal.reshape(self.re_shape)
    
    
def ashift(array):
    return np.fft.fftshift(np.abs(array))

def quantify_aliasing_noise(sig, dec_fact, dt=1, numtaps_fact=41, nyq_rat=2.5 , do_plot=False):
    import scipy.signal
    
    N=sig.size
    fs=1/dt
    
    dt_dec=dt*dec_fact
    N_dec = int(np.floor(N/dec_fact)) 
    # ceil would also work, but breaks indexing for aliasing noise estimation
    # with floor though, care must be taken to shorten the time domain signal to N_dec full blocks before slicing
    
    cutoff=fs/nyq_rat/dec_fact
    numtaps= numtaps_fact*dec_fact
    
    t = np.linspace(0,(N-1)*dt,N)
    
    # compute spectrum
    if do_plot:
        fft_sig = np.fft.fft(sig)#/np.sqrt(N)

    # fir lowpass filter
    # fir_firwin = scipy.signal.firwin(numtaps, cutoff/fs*2)
    fir_firwin = scipy.signal.firwin(numtaps, cutoff,fs=fs)

    #filter signal
    sig_filt = np.convolve(sig, fir_firwin, 'same')

    fft_filt = np.fft.fft(sig_filt)#/np.sqrt(N)
    fft_filt_alias = np.copy(fft_filt)


    #decimate signal
    sig_filt_dec = np.copy(sig_filt[0:N_dec*dec_fact:dec_fact])
    # correct for power loss due to decimation
    # https://en.wikipedia.org/wiki/Downsampling_(signal_processing)#Anti-aliasing_filter
    sig_filt_dec*=dec_fact

    #fft_dec = np.fft.fft(sig_dec)#/np.sqrt(N_dec)
    fft_filt_dec = np.fft.fft(sig_filt_dec)#/np.sqrt(N_dec)

    t_dec = t[0::dec_fact]
    if do_plot:
        plt.figure()
        freq=np.fft.fftshift(np.fft.fftfreq(N,d=dt))
        plt.semilogy(freq, ashift(fft_sig), label='original signal', color='whitesmoke')
        plt.semilogy(freq, ashift(fft_filt), label='filtered signal', color='lightgrey')

    fft_filt_alias = np.zeros(N_dec,dtype=complex)
    fft_filt_non_alias = np.zeros(N_dec,dtype=complex)

    fft_filt_non_alias[:N_dec//2] += np.copy(fft_filt[:1*N_dec//2])
    fft_filt_non_alias[N_dec//2:] += np.copy(fft_filt[-1*N_dec//2:])

    pos_alias = fft_filt_alias[:N_dec//2]
    neg_alias = fft_filt_alias[N_dec//2:]

    for i in reversed(range(1,dec_fact)):
        slp=slice(i*N_dec//2,(i+1)*N_dec//2)
        sln=slice(-(i+1)*N_dec//2,-i*N_dec//2)

        if i%2:# alias and fold
            neg=fft_filt[slp]
            pos=fft_filt[sln]
            
            if do_plot:
                ls='dashed'
                freqp=np.fft.fftfreq(N,d=dt)[sln]
                freqn=np.fft.fftfreq(N,d=dt)[slp]
        else: # alias
            pos=fft_filt[slp]
            neg=fft_filt[sln]
            
            if do_plot:
                ls='dotted'
                freqp=np.fft.fftfreq(N,d=dt)[slp]
                freqn=np.fft.fftfreq(N,d=dt)[sln]

        pos_alias+=pos
        neg_alias+=neg
        if do_plot:
            plt.semilogy(freqp, np.abs(pos), color='k', ls=ls)#,marker='x')
            plt.semilogy(freqn, np.abs(neg), color='k', ls=ls)#,marker='x')
            plt.semilogy(np.fft.fftshift(np.fft.fftfreq(N_dec,dt_dec)),np.copy(ashift(fft_filt_alias)), color='k', ls=ls)
    else:
        if do_plot:
            plt.semilogy(np.fft.fftshift(np.fft.fftfreq(N_dec,dt_dec)),np.copy(ashift(fft_filt_alias+fft_filt_non_alias)), color='k', label='alias')
    
    p_fft_filt_alias = np.mean(np.abs(fft_filt_alias/np.sqrt(N_dec))**2)
    p_fft_filt_non_alias = np.mean(np.abs(fft_filt_non_alias/np.sqrt(N_dec))**2)
    snr_alias = p_fft_filt_non_alias/p_fft_filt_alias

    if do_plot:
        sig_filt_alias = np.fft.ifft(fft_filt_alias)
        sig_filt_nonalias = np.fft.ifft(fft_filt_non_alias)

        # signal powers
        p_sig = np.mean(np.abs(sig)**2)
        p_fft =np.mean(np.abs(fft_sig/np.sqrt(N))**2)
        p_sig_filt = np.mean(np.abs(sig_filt)**2)
        p_fft_filt = np.mean(np.abs(fft_filt/np.sqrt(N))**2)
        p_sig_filt_dec = np.mean(np.abs(sig_filt_dec)**2)
        p_fft_filt_dec = np.mean(np.abs(fft_filt_dec/np.sqrt(N_dec))**2)
        p_sig_filt_alias = np.mean(np.abs(sig_filt_alias)**2)
        p_fft_filt_alias = np.mean(np.abs(fft_filt_alias/np.sqrt(N_dec))**2)
        p_sig_filt_non_alias = np.mean(np.abs(sig_filt_nonalias)**2)
        p_fft_filt_non_alias = np.mean(np.abs(fft_filt_non_alias/np.sqrt(N_dec))**2)
        print('Power signal: ',p_sig)
        print('Power filtered signal: ',p_sig_filt, 'filtered spectrum: ', p_fft_filt)
        print('Power decimated signal: ',p_sig_filt_dec)
        print('Power alias spectrum: ',p_fft_filt_alias)
        print('Power non-alias spectrum: ',p_fft_filt_non_alias)
        print('SNR alias: ', snr_alias, ' in dB: ', np.log10(snr_alias)*10)

        freq_dec=np.fft.fftshift(np.fft.fftfreq(N_dec,d=dt_dec))
        plt.semilogy(freq_dec, ashift(fft_filt_dec), ls='dashed', color='darkgrey', label='filtered decimated')

        plt.xlim(xmin=0,xmax=fs/2)
        plt.axvline(fs/2/dec_fact,color='k')
        plt.axvline(cutoff,ls='dashed',color='k')
        plt.legend()
    
    return sig_filt_dec, snr_alias

def system_signal_decimate(N,dec_fact, numtap_fact, nyq_rat, do_plot=False, **kwargs):
    fmax=400
    df=fmax/(N//2+1)
    dt=1/df/N
    fs=1/dt
    #signal processing parameters
    #dec_fact=4
    #dec_fact=int(inputs[i,1])

    # allocation
    omegas = np.linspace(0,fmax,N//2+1,False)*2*np.pi
    t = np.linspace(0,(N-1)*dt,N)
    frf= np.zeros((N//2+1,), dtype=complex)
    assert df*2*np.pi == (omegas[-1]-omegas[0])/(N//2+1-1)

    # system parameters
    L=200
    E=2.1e11
    rho=7850
    num_modes=int(np.floor((fmax*2*np.pi*np.sqrt(rho/E)*L/np.pi*2+1)/2))

    #system generation
    ks=np.random.rand(num_modes)+0.5
    omegans = (2*np.arange(1,num_modes+1)-1)/2*np.pi/L*np.sqrt(E/rho)
    zetas = np.zeros_like(omegans)
    zetas[:]=0.001
    for k,omegan, zeta in zip(ks,omegans, zetas):
        frf+=1/(k*(1+2*1j*zeta*omegas/omegan-(omegas/omegan)**2))

    # random input    
    phase=(np.random.rand(N//2+1)-0.5)
    Pomega = np.ones_like(frf)*np.exp(1j*phase*2*np.pi)

    # time domain signal
    sig=np.fft.irfft(frf*Pomega)
    
    return (np.log10(quantify_aliasing_noise(sig,dec_fact,dt,numtap_fact,nyq_rat,do_plot)[1])*10,)

def uq_dec_noise():
    import seaborn
    import sys
    sys.path.append('/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/uncertainty')
    import data_manager
    import os
    
    num_samples = 10
    
    title = 'decimation_noise'
    savefolder = '/vegas/scratch/womo1998/modal_uq/'
    result_dir = '/vegas/scratch/womo1998/modal_uq/'
    
    if not os.path.exists(result_dir + title + '.nc') or False:
        dm = data_manager.DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir=result_dir,
                                   overwrite=True)
        dm.generate_sample_inputs(names=['N',
                                                   'dec_fact',
                                                   'numtap_fact',
                                                   'nyq_rat'],
                                            distributions=[('choice', [2**np.arange(5, 18)]),
                                                           ('integers', (2, 15)),
                                                           ('integers', (5, 61)),
                                                           ('uniform', (2, 4)),
                                                           ],
                                            num_samples=num_samples)
    
        dm.post_process_samples(db='in', names=['N','dec_fact','numtap_fact','nyq_rat'], )
    elif False:
        # add_samples
        dm = data_manager.DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        dm.enrich_sample_set(total_samples=5000)
    
    elif False:
        #evaluate input samples
        dm = data_manager.DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        dm.evaluate_samples(func=system_signal_decimate, 
                                      arg_vars={'N':'N','dec_fact':'dec_fact',
                                                'numtap_fact':'numtap_fact','nyq_rat':'nyq_rat'},
                                      ret_names=['snrs'],
                                      chwdir=True, dry_run=False)
    elif True:
        dm = data_manager.DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        dm.post_process_samples(db='merged',scales = ['log','linear','linear','linear','linear'])
        dm.clear_locks()
        
        
def main():
    from model import mechanical
    
    skip_existing = True
    working_dir = '/dev/shm/womo1998/'
    result_dir = '/vegas/scratch/womo1998/modal_uq/'
    jid = 'filter_example'
    #jid='acquire_example'
    
    ansys = mechanical.Mechanical.start_ansys(working_dir, jid)
    
    num_nodes = 21
    damping = 0.01
    num_modes = 20
    dt_fact = 0.01
    num_cycles = 10
    meas_nodes = list(range(1, 22))
    
    # load/generate  mechanical object:
    if True:
        # ambient response of a 20 dof rod,
        f_scale = 1000
        d0 = None
        savefolder = os.path.join(result_dir, jid, 'ambient')
    else:
        # impulse response of a 20 dof rod
        f_scale = None
        d0 = 1
        savefolder = os.path.join(result_dir, jid, 'impulse')
        
    if not os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')) or not skip_existing:
        mech = mechanical.generate_mdof_time_hist(ansys=ansys, num_nodes=num_nodes, damping=damping,
                                                  num_modes=num_modes, d0=d0, f_scale=f_scale,
                                                  dt_fact=dt_fact, num_cycles=num_cycles,
                                                  meas_nodes=meas_nodes,)
        mech.save(savefolder)
    else:
        mech = mechanical.Mechanical.load(jid, savefolder)
                
    # pass mechanical object to acquire
    acqui = Acquire(mech)
    
    if True:
        # pass to PreProcessingTools and create filter example with it
        from core import PreprocessingTools
        figt, axest = plt.subplots(2, 2, sharex=True, sharey=True)
        axest = axest.flat
        figf, axesf = plt.subplots(2, 2, sharex=True, sharey=True)
        axesf = axesf.flat
        for filt, order, cutoff, ax1, ax2 in [('moving_average', 5, 58.19, axest[0], axesf[0]),
                                                ('brickwall', 5, 58.19, axest[1], axesf[1]),
                                                ('butterworth', 5, 58.19, axest[2], axesf[2]),
                                                ('cheby1', 5, 58.19, axest[3], axesf[3])]:
            
            prep_data = PreprocessingTools.PreprocessData(**acqui.to_preprocessdata())
            prep_data.filter_data(filt,order,cutoff)
            prep_data.plot_data(ax1)
            prep_data.plot_psd(ax2)
        
        
        
        
    else:
        # apply sensor frf
        acqui.apply_FRF(type)
        
        # add sensor / cabling  noise
        acqui.add_noise(snr_db, noise_power)
        
        # decimate and estimate decimation noise / quantization noise
        acqui.sample(f_max, fs_factor)
        
        prep_data = PreprocessingTools.PreprocessData(**acqui.to_preprocessdata())
        
        nodes, lines, chan_dofs = mech.get_geometry()
        
        geometry = PreprocessingTools.GeometryProcessor(nodes, lines)
        
    
    # 
        
if __name__ == '__main__':
    # N=8192
    # dec_fact=4
    # numtap_fact=41
    # nyq_rat=2.5
    # system_signal_decimate(N,dec_fact, numtap_fact,nyq_rat, True)
    # plt.show()
    # uq_dec_noise()
    main()
    
