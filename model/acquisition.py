'''
models the signal acquisition:


'''

import numpy as np
#import scipy.signal
import matplotlib.pyplot as plt


class Acquire(object):

    def __init__(self, t_vals, signal=None, IRF_matrix=None):
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
        
        assert self.num_timesteps == len(t_vals)
        
        self.t_vals = t_vals
        self.signal = signal
        
    
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
        
if __name__ == '__main__':
    # N=8192
    # dec_fact=4
    # numtap_fact=41
    # nyq_rat=2.5
    # system_signal_decimate(N,dec_fact, numtap_fact,nyq_rat, True)
    # plt.show()
    uq_dec_noise()
    
