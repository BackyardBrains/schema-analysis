import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.stats import linregress, sem
from scipy import signal

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'arm_angle')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
RESULTS_DIR = os.path.join(ROOT, 'results', 'arm_angle')

def load_events(events_file):
    events = []
    with open(events_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                marker = parts[0].strip()
                time_s = float(parts[1].strip())
                events.append((marker, time_s))
    return events

def get_voltage_at_time(data, sample_rate, time_s, window_s=0.2):
    idx_center = int(time_s * sample_rate)
    window_samples = int(window_s * sample_rate / 2)
    start_idx = max(0, idx_center - window_samples)
    end_idx = min(len(data), idx_center + window_samples)
    
    snippet = data[start_idx:end_idx]
    if len(snippet.shape) == 1:
        return np.median(snippet), np.median(snippet)
    else:
        return np.median(snippet[:, 0]), np.median(snippet[:, 1])

def process_subject(subject, wav_file, events_file, calib_angles):
    sample_rate, data = wavfile.read(wav_file)
    events = load_events(events_file)
    
    calib_events = {}
    for marker, t in events:
        if marker in ['1', '2', '3', '4', '5']:
            if marker not in calib_events:
                calib_events[marker] = t
        if marker == '8':
            break

    ch0_volts = []
    ch1_volts = []
    angles = []
    
    for marker in ['1', '2', '3', '4', '5']:
        if marker in calib_events and calib_angles[int(marker)-1] is not None and not np.isnan(calib_angles[int(marker)-1]):
            t = calib_events[marker]
            v0, v1 = get_voltage_at_time(data, sample_rate, t)
            ch0_volts.append(v0)
            ch1_volts.append(v1)
            angles.append(calib_angles[int(marker)-1])
            
    if len(angles) < 2:
        return None
        
    slope0, intercept0, r0, p0, std_err0 = linregress(ch0_volts, angles)
    slope1, intercept1, r1, p1, std_err1 = linregress(ch1_volts, angles)
    
    if len(data.shape) > 1:
        ch0_data = data[:, 0]
        ch1_data = data[:, 1]
    else:
        ch0_data = data
        ch1_data = data

    times = np.arange(len(ch0_data)) / sample_rate
    ch0_angles = slope0 * ch0_data + intercept0
    ch1_angles = slope1 * ch1_data + intercept1
    
    error_series = np.abs(ch0_angles - ch1_angles)
    
    follow_markers = [t for m, t in events if m == '8']
    vib_markers = [t for m, t in events if m == '9']
    
    t_grid = np.linspace(-1, 25, num=2600)
    
    follow_err = None
    vib_err = None
    r_follow = np.nan
    r_vib = np.nan
    
    if len(follow_markers) >= 2:
        t0 = follow_markers[0]
        t1 = follow_markers[1]
        
        # Extracted bounded mask for XCorr
        mask = (times >= t0) & (times <= t1)
        if np.sum(mask) > 10:
            c0 = ch0_angles[mask]
            c1 = ch1_angles[mask]
            
            # Find optimal lag using cross-correlation 
            # to adjust for natural human tracking reflex delay
            corr = signal.correlate(c0 - np.mean(c0), c1 - np.mean(c1), mode='full')
            lags = signal.correlation_lags(len(c0), len(c1), mode='full')
            
            # Constrain lag to max 2.5 seconds
            max_lag_samples = int(2.5 * sample_rate)
            valid_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
            
            best_lag = 0
            if np.any(valid_mask):
                valid_lags = lags[valid_mask]
                valid_corr = corr[valid_mask]
                best_lag = valid_lags[np.argmax(valid_corr)]
                
            # Shift the entire signals by best_lag for purely time-aligned error calculation
            if best_lag > 0:
                c0_eff = ch0_angles[best_lag:]
                c1_eff = ch1_angles[:-best_lag]
                t_eff = times[best_lag:]
            elif best_lag < 0:
                c0_eff = ch0_angles[:best_lag]
                c1_eff = ch1_angles[-best_lag:]
                t_eff = times[-best_lag:]
            else:
                c0_eff = ch0_angles
                c1_eff = ch1_angles
                t_eff = times
                
            # Re-crop aligned mask for correlation calculation (10 to 25s window)
            mask_eff = (t_eff >= t0 + 10) & (t_eff <= min(t0 + 25, t1))
            if np.sum(mask_eff) > 10:
                c0_aligned = c0_eff[mask_eff]
                c1_aligned = c1_eff[mask_eff]
                if np.std(c0_aligned) > 0 and np.std(c1_aligned) > 0:
                    r_follow = np.corrcoef(c0_aligned, c1_aligned)[0, 1]
            
            # Recompute error specifically for Follow Me using the time-aligned versions
            aligned_error = np.abs(c0_eff - c1_eff)
            if t0 - 1 >= 0 and t0 + 25 <= t_eff[-1]:
                rel_times = t_eff - t0
                follow_err = np.interp(t_grid, rel_times, aligned_error)

    if len(vib_markers) >= 2:
        t0 = vib_markers[0]
        t1 = vib_markers[1]
        
        # 10 to 25s window for vibration correlation
        mask_corr = (times >= t0 + 10) & (times <= min(t0 + 25, t1))
        if np.sum(mask_corr) > 10:
            c0 = ch0_angles[mask_corr]
            c1 = ch1_angles[mask_corr]
            if np.std(c0) > 0 and np.std(c1) > 0:
                r_vib = np.corrcoef(c0, c1)[0, 1]
                
        if t0 - 1 >= 0 and t0 + 25 <= times[-1]:
            rel_times = times - t0
            
            c0_zeroed = ch0_angles - np.interp(0, rel_times, ch0_angles)
            c1_zeroed = ch1_angles - np.interp(0, rel_times, ch1_angles)
            
            # Sensor 1 (ch0) is ALWAYS the stimulated/stationary arm.
            # Sensor 2 (ch1) is ALWAYS the tracking arm.
            stat_series = np.abs(c0_zeroed)
            track_series = np.abs(c1_zeroed)
                
            track_err = np.interp(t_grid, rel_times, track_series)
            stat_err = np.interp(t_grid, rel_times, stat_series)
            vib_err = (track_err, stat_err)

    return follow_err, vib_err, r_follow, r_vib

def run_group_analysis():
    print("Starting Group Arm Angle Analysis...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    json_path = os.path.join(DATA_DIR, 'metadata.json')
    if not os.path.exists(json_path):
        print("Metadata JSON not found.")
        return
        
    with open(json_path, 'r') as f:
        metadata = json.load(f)
        
    wav_files = glob.glob(os.path.join(RAW_DIR, "*.wav"))
    
    all_follow = []
    all_vib_track = []
    all_vib_stat = []
    corr_follow = []
    corr_vib = []
    subject_names = []
    
    for wav_file in wav_files:
        basename = os.path.basename(wav_file).replace('.wav', '')
        parts = basename.split('_')
        if len(parts) >= 4:
            subject = parts[-1]
            events_file = os.path.join(RAW_DIR, f"{basename}-events.txt")
            
            if os.path.exists(events_file):
                if subject in metadata:
                    info = metadata[subject]
                    if info.get('exclude'):
                        pass
                    else:
                        vividness = info.get('vividness', 0)
                        if vividness < 3:
                            continue
                            
                        calib_angles = [float(a) if a is not None else np.nan for a in info.get('angles', [])]
                        res = process_subject(subject, wav_file, events_file, calib_angles)
                        if res is not None:
                            f_err, v_err, r_f, r_v = res
                            if f_err is not None: all_follow.append(f_err)
                            if v_err is not None:
                                all_vib_track.append(v_err[0])
                                all_vib_stat.append(v_err[1])
                            # Ensure paired analysis for correlations
                            if not np.isnan(r_f) and not np.isnan(r_v):
                                subject_names.append(subject)
                                corr_follow.append(r_f)
                                corr_vib.append(r_v)
                            print(f"Added {subject} to group analysis.")
                else:
                    print(f"Subject {subject} not found in metadata.")
    
    t_grid = np.linspace(-1, 25, num=2600)
    
    def plot_group(data_list, title, ylabel, filename):
        if not data_list:
            print(f"No data for {title}")
            return
            
        data_matrix = np.vstack(data_list)
        mean = np.mean(data_matrix, axis=0)
        st_err = sem(data_matrix, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_grid, mean, color='#D55E00', linewidth=2, label='Group Average')
        plt.fill_between(t_grid, mean - st_err, mean + st_err, color='#D55E00', alpha=0.3, label='Standard Error')
        
        plt.axvline(0, color='black', linestyle='--', alpha=0.6, label='Stimulation/Phase Start')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved {save_path} (N={len(data_list)})")
        
    def plot_group_dual(track_list, stat_list, title, ylabel, filename):
        if not track_list or not stat_list:
            return
            
        t_mat = np.vstack(track_list)
        s_mat = np.vstack(stat_list)
        
        t_mean = np.mean(t_mat, axis=0)
        t_err = sem(t_mat, axis=0)
        s_mean = np.mean(s_mat, axis=0)
        s_err = sem(s_mat, axis=0)
        
        plt.figure(figsize=(10, 6))
        # Tracking arm
        plt.plot(t_grid, t_mean, color='#C44E52', linewidth=2, label='Subject Matching Arm (Physical Movement)')
        plt.fill_between(t_grid, t_mean - t_err, t_mean + t_err, color='#C44E52', alpha=0.3)
        # Stationary arm
        plt.plot(t_grid, s_mean, color='#4C72B0', linewidth=2, label='Manipulated Arm (Stationary)')
        plt.fill_between(t_grid, s_mean - s_err, s_mean + s_err, color='#4C72B0', alpha=0.3)
        
        plt.axvline(0, color='black', linestyle='--', alpha=0.6, label='Stimulation Start')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved {save_path} (N={len(track_list)})")
        
    plot_group(all_follow, "Group Average 'Follow Me' Tracking Error (High Vividness)", "Absolute Arm Difference (Degrees)", "group_follow_me.png")
    plot_group_dual(all_vib_track, all_vib_stat, "Group Average Illusion Drift (Vibration Phase)", "Absolute Displacement From Baseline (Degrees)", "group_vibration.png")
    
    # Plot correlation bar chart
    if corr_follow and corr_vib:
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(corr_follow, corr_vib)
        print(f"Paired T-Test Results: t={t_stat:.3f}, p={p_val:.5f}")
        
        plt.figure(figsize=(8, 6))
        means = [np.mean(corr_follow), np.mean(corr_vib)]
        errs = [sem(corr_follow), sem(corr_vib)]
        
        bars = plt.bar(['Follow Me Phase\n(10-25s Window)', 'Vibration Phase\n(10-25s Window)'], means, yerr=errs, capsize=10, color=['#4C72B0', '#C44E52'])
        plt.axhline(0, color='black', linewidth=1)
        plt.ylim(-0.5, 1.2)
        plt.ylabel("Pearson Correlation Coefficient (r)")
        plt.title("Sensor Coordination: Follow Me vs. Vibration", fontweight='bold')
        
        # Add N above bars
        plt.text(0, means[0] + errs[0] + 0.05, f"N={len(corr_follow)}", ha='center')
        plt.text(1, means[1] + errs[1] + 0.05, f"N={len(corr_vib)}", ha='center')
        
        # Add significance bracket
        h = max(means[0] + errs[0], means[1] + errs[1]) + 0.15
        asterisks = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        plt.plot([0, 0, 1, 1], [h-0.05, h, h, h-0.05], lw=1.5, c='black')
        plt.text(0.5, h + 0.02, f"p = {p_val:.4f} {asterisks}", ha='center', va='bottom', color='black', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "group_correlation.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved {save_path}")
        
    if subject_names:
        # 1. Individual Plot for Follow Me
        plt.figure(figsize=(10, 6))
        plt.bar(subject_names, corr_follow, color='#4C72B0')
        plt.axhline(0, color='black', linewidth=1)
        plt.axhline(np.mean(corr_follow), color='r', linestyle='--', label=f'Mean (r={np.mean(corr_follow):.2f})')
        plt.ylim(-1.0, 1.1)
        plt.ylabel("Pearson Correlation Coefficient (r)")
        plt.title("Individual Sensor Coordination: Follow Me Phase (10-25s)", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path_f = os.path.join(RESULTS_DIR, "individual_corr_follow.png")
        plt.savefig(save_path_f, dpi=200)
        plt.close()
        
        # 2. Individual Plot for Vibration
        plt.figure(figsize=(10, 6))
        plt.bar(subject_names, corr_vib, color='#C44E52')
        plt.axhline(0, color='black', linewidth=1)
        plt.axhline(np.mean(corr_vib), color='r', linestyle='--', label=f'Mean (r={np.mean(corr_vib):.2f})')
        plt.ylim(-1.0, 1.1)
        plt.ylabel("Pearson Correlation Coefficient (r)")
        plt.title("Individual Sensor Coordination: Vibration Phase (10-25s)", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path_v = os.path.join(RESULTS_DIR, "individual_corr_vibration.png")
        plt.savefig(save_path_v, dpi=200)
        plt.close()
        print(f"Saved Individual plots")

if __name__ == '__main__':
    run_group_analysis()
