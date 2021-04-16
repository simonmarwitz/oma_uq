'''
models the signal acquisition:


'''


import numpy as np

class Acquire(object):

    def __init__(self,t_vals, signal=None, IRF_matrix=None):
        '''
        input a synthethised signal (ambient response, impulse response) and corresponding t_vals 
        '''

        pass
    
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
        
        array = array[1::6, :]
        dt = (array[-1, 0] - array[0, 0]) / (array.shape[0] - 1)
    
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
    
    def get_signal(self, IRF_matrix=False):
        if IRF_matrix:
            return self.t_vals, self.IRF_matrix
        else:
            return self.t_vals, self.signal
        
if __name__ == '__main__':
    pass