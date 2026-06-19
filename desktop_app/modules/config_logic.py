
import math

class RadarConfigLogic:
    def __init__(self):
        # Constants for xWR68xx
        self.c = 3e8
        self.start_freq = 60e9 # 60 GHz
        self.lambda_ = self.c / self.start_freq # 0.005 m
        
        # Hardware Limits
        self.max_bandwidth = 3.9e9 # 3.9 GHz (Safe margin from 4GHz)
        self.max_sample_rate = 6250e3 # 6.25 Msps (example)
        self.max_slope = 100e12 # 100 MHz/us
        self.adc_samples = 256 # Fixed for simplicity usually
        self.num_tx = 2 # 2Tx TDM assumed
        
        # Visualization Targets (Initial)
        self.target_range_res = 0.044
        self.target_max_range = 9.0
        self.target_max_vel = 1.0
        self.target_vel_res = 0.13
        
    def calculate_from_inputs(self, range_res, max_range, max_vel):
        """
        Derive profileCfg parameters from high-level targets with determining physics limits.
        Refined to match xWR6843 constraints:
        - Max Bandwidth: 4 GHz
        - Max Slope: 100 MHz/us
        - Max Fs: 12.5 Msps (approx half of 25Msps for complex 1x)
        """
        # 1. Bandwidth (B) determined by Range Resolution
        # range_res = c / (2 * B)  => B = c / (2 * range_res)
        bandwidth = self.c / (2 * range_res)
        
        # Clamp Bandwidth
        if bandwidth > self.max_bandwidth: 
            bandwidth = self.max_bandwidth
            # Recalculate range_res if clamped? User slider usually sets target.
            range_res = self.c / (2 * bandwidth)
            
        # 2. Number of ADC Samples (N) determined by Max Range
        # R_max = N * range_res
        # Find smallest Power of 2 N such that N * range_res >= max_range
        # Valid: 64, 128, 256, 512, 1024
        
        # [FIX] Add tolerance (0.95) to prevent jumping to next power of 2 
        # when Max Range slider is set to theoretical limit of N=256.
        min_samples_needed = (max_range * 0.95) / range_res
        needed_samples = 64
        while needed_samples < min_samples_needed and needed_samples < 1024:
            needed_samples *= 2
            
        actual_max_range = needed_samples * range_res
        
        # 3. Sampling Rate (Fs) and Slope (S)
        # B = S * T_active = S * (N / Fs)
        # S = (B * Fs) / N
        # Constraint 1: S <= MaxSlope (100 MHz/us)
        # Constraint 2: Fs <= MaxFs (6.25 Msps typical for xWR6843 complex 1x streaming)
        # Note: If N doubles, and Fs is maxed, T_ramp doubles, V_max halves.
        
        s_max_hz = self.max_slope # 100e12
        
        # Standard TI Sampling Rates (ksps)
        # Based on divisibility of master clock.
        valid_fs_gamut = [
            1000, 1125, 1250, 1500, 1600, 1875, 2000, 2250, 2500, 
            3000, 3200, 3750, 4000, 4500, 5000, 6000, 6250
        ]
        valid_fs_gamut = [f * 1e3 for f in valid_fs_gamut] # Convert to Hz

        # Calculate max Fs allowed by Slope
        # Fs <= (S_max * N) / B
        slope_limited_fs = (s_max_hz * needed_samples) / bandwidth
        
        # Hard cap at max hardware rate
        effective_max_fs = min(self.max_sample_rate, slope_limited_fs)
        
        # Find largest valid Fs <= effective_max_fs
        target_fs = valid_fs_gamut[0]
        for fs_val in valid_fs_gamut:
            if fs_val <= effective_max_fs:
                target_fs = fs_val
            else:
                break
        
        # Calculate resulting Slope
        # S = B * Fs / N
        slope = (bandwidth * target_fs) / needed_samples
        
        
        # 4. Chirp Time (Tc) determined by Max Vel
        # V_max = lambda / (4 * Tc_total_per_chirp_sequence) (?)
        # For TDM MIMO (2Tx):
        # The phase shift is measured between chirps of the SAME antenna in subsequent frames? 
        # No, between chirps of same antenna in SAME frame (Doppler).
        # T_c_effective = NumTx * Tc_physical (time between samples of same Tx-Rx pair)
        # V_max = lambda / (4 * T_c_effective) = lambda / (4 * NumTx * Tc_physical)
        
        # 4. Chirp Time (Tc) Optimization
        # Priority: Maximize V_max (Shortest Tc) to ensure stability and best anti-aliasing.
        # Ignoring user 'max_vel' input for Tc calculation to prevent large Idle Times.
        
        # Constraint: Tc_physical >= T_ramp + T_idle
        t_sampling = needed_samples / target_fs
        min_idle = 10e-6 # 10us (Robust margin for IWR6843 stability)
        adc_start_time = 7e-6 # 7us
        ramp_excess_time = 5e-6 # 5us (Robust margin)
        
        # Minimum physical chirp time required to fit the ramp
        min_tc_phys = min_idle + adc_start_time + t_sampling + ramp_excess_time
        
        # [FIX] Always use minimal Tc to maximize Max Velocity and stability
        needed_tc_phys = min_tc_phys 
            
        actual_max_vel = self.lambda_ / (4 * needed_tc_phys * self.num_tx)
        
        return {
            "bandwidth": bandwidth,
            "range_res": range_res,
            "samples": needed_samples,
            "fs": target_fs,
            "slope": slope,
            "max_range": actual_max_range,
            "tc": needed_tc_phys,
            "max_vel": actual_max_vel
        }

    def calculate_rcs_metrics(self, desired_rcs, max_unambig_range):
        """
        Calculate RCS related metrics based on Radar Equation.
        Reference: High-performance EVM ~56.5m range for 1sqm target (derived from user data).
        R_max = R_ref * (sigma / 1.0)^0.25
        sigma_min = 1.0 * (R_unambig / R_ref)^4
        """
        r_ref = 56.5 # m for 1sqm
        
        # 1. Max Range for Desired RCS
        # R = Ref * sigma^0.25
        max_range_for_rcs = r_ref * math.pow(desired_rcs, 0.25)
        
        # 2. RCS at Max Unambiguous Range
        # sigma = (R / Ref)^4
        if max_unambig_range > 0:
            rcs_at_max_range = math.pow(max_unambig_range / r_ref, 4)
        else:
            rcs_at_max_range = 0.0
            
        return {
            "max_range_for_rcs": max_range_for_rcs,
            "rcs_at_max_unambig": rcs_at_max_range
        }
