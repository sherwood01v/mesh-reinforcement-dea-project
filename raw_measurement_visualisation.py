import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from scipy.signal import savgol_filter
from scipy.stats import zscore
import numpy as np

# Actuator dimensions
ACTUATOR_WIDTH = 30  # mm
ACTUATOR_HEIGHT = 70  # mm

# Threshold for ON/OFF voltage detection
VOLTAGE_THRESHOLD = 0.5  # kV


def detect_on_off_periods(voltages, times, displacements, threshold=0.5):
    """
    Detect voltage ON and OFF periods and calculate steady-state displacements.
    
    Parameters:
    -----------
    voltages : array
        Voltage signal
    times : array
        Time array
    displacements : array
        Displacement signal
    threshold : float
        Voltage threshold to distinguish ON from OFF (kV)
    
    Returns:
    --------
    periods_data : list of dict
        List containing data for each period with ON/OFF state and mean displacement
    """
    # Binarize voltage signal
    voltage_binary = (voltages > threshold).astype(int)
    
    # Find transitions
    diff = np.diff(voltage_binary)
    rising_edges = np.where(diff == 1)[0] + 1  # OFF to ON
    falling_edges = np.where(diff == -1)[0] + 1  # ON to OFF
    
    # Determine starting state
    if len(rising_edges) == 0 and len(falling_edges) == 0:
        # No transitions - entire signal is one state
        if voltage_binary[0] == 1:
            return [{'period': 1, 'state': 'ON', 'start_idx': 0, 'end_idx': len(voltages)-1, 
                    'mean_displacement': np.mean(displacements), 'is_good': True}]
        else:
            return [{'period': 1, 'state': 'OFF', 'start_idx': 0, 'end_idx': len(voltages)-1,
                    'mean_displacement': np.mean(displacements), 'is_good': True}]
    
    # Build list of all edges with their type
    edges = []
    for edge in rising_edges:
        edges.append((edge, 'rising'))
    for edge in falling_edges:
        edges.append((edge, 'falling'))
    edges.sort(key=lambda x: x[0])
    
    # Extract periods
    periods_data = []
    period_num = 1
    
    # Handle initial period before first edge
    if len(edges) > 0:
        first_edge_idx = edges[0][0]
        if first_edge_idx > 100:  # Only consider if period is long enough
            initial_state = 'OFF' if edges[0][1] == 'rising' else 'ON'
            mean_disp = np.mean(displacements[:first_edge_idx])
            periods_data.append({
                'period': period_num,
                'state': initial_state,
                'start_idx': 0,
                'end_idx': first_edge_idx,
                'mean_displacement': mean_disp,
                'is_good': True
            })
            period_num += 1
    
    # Extract periods between edges
    for i in range(len(edges) - 1):
        start_idx = edges[i][0]
        end_idx = edges[i+1][0]
        
        # Determine state based on the edge type we just crossed
        state = 'ON' if edges[i][1] == 'rising' else 'OFF'
        
        # Only consider periods with reasonable length
        if end_idx - start_idx > 100:
            mean_disp = np.mean(displacements[start_idx:end_idx])
            periods_data.append({
                'period': period_num,
                'state': state,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'mean_displacement': mean_disp,
                'is_good': True
            })
            period_num += 1
    
    # Handle final period after last edge
    if len(edges) > 0:
        last_edge_idx = edges[-1][0]
        if len(voltages) - last_edge_idx > 100:
            final_state = 'ON' if edges[-1][1] == 'rising' else 'OFF'
            mean_disp = np.mean(displacements[last_edge_idx:])
            periods_data.append({
                'period': period_num,
                'state': final_state,
                'start_idx': last_edge_idx,
                'end_idx': len(voltages)-1,
                'mean_displacement': mean_disp,
                'is_good': True
            })
    
    return periods_data


def detect_outliers_and_calculate_delta_d(periods_data, z_threshold=2.0):
    """
    Detect outlier periods and calculate steady-state displacement.
    
    Parameters:
    -----------
    periods_data : list of dict
        List of period data from detect_on_off_periods
    z_threshold : float
        Z-score threshold for outlier detection
    
    Returns:
    --------
    delta_d : float
        Mean steady-state displacement (ON - OFF)
    periods_data : list of dict
        Updated periods_data with is_good flags
    stats : dict
        Statistics about the calculation
    """
    # Separate ON and OFF periods
    on_periods = [p for p in periods_data if p['state'] == 'ON']
    off_periods = [p for p in periods_data if p['state'] == 'OFF']
    
    if len(on_periods) == 0 or len(off_periods) == 0:
        return None, periods_data, {'error': 'Not enough ON/OFF periods'}
    
    # Detect outliers in ON periods using z-score if we have enough data
    if len(on_periods) >= 3:
        on_displacements = np.array([p['mean_displacement'] for p in on_periods])
        on_z_scores = np.abs(zscore(on_displacements))
        for i, period in enumerate(on_periods):
            if on_z_scores[i] > z_threshold:
                period['is_good'] = False
    
    # Detect outliers in OFF periods
    if len(off_periods) >= 3:
        off_displacements = np.array([p['mean_displacement'] for p in off_periods])
        off_z_scores = np.abs(zscore(off_displacements))
        for i, period in enumerate(off_periods):
            if off_z_scores[i] > z_threshold:
                period['is_good'] = False
    
    # Calculate mean using only good periods
    good_on_periods = [p for p in on_periods if p['is_good']]
    good_off_periods = [p for p in off_periods if p['is_good']]
    
    if len(good_on_periods) == 0 or len(good_off_periods) == 0:
        return None, periods_data, {'error': 'No good periods after outlier removal'}
    
    mean_on = np.mean([p['mean_displacement'] for p in good_on_periods])
    mean_off = np.mean([p['mean_displacement'] for p in good_off_periods])
    delta_d = mean_on - mean_off
    
    stats = {
        'total_on_periods': len(on_periods),
        'total_off_periods': len(off_periods),
        'good_on_periods': len(good_on_periods),
        'good_off_periods': len(good_off_periods),
        'mean_on_displacement': mean_on,
        'mean_off_displacement': mean_off,
        'std_on_displacement': np.std([p['mean_displacement'] for p in good_on_periods]),
        'std_off_displacement': np.std([p['mean_displacement'] for p in good_off_periods]),
        'delta_d': delta_d
    }
    
    return delta_d, periods_data, stats


# Get all CSV files
# csv_files = glob.glob('*.csv')

# Ask user for the directory containing the CSV files
csv_dir = input("Enter the directory where CSV files are stored (leave empty for current directory): ").strip()
csv_dir = csv_dir if csv_dir else '.'
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# Optionally process only specific file(s) - uncomment to use
# csv_files = [csv_files[0]]  # Process only first file

# Create a directory for the plots if it doesn't exist
if not os.path.exists('final_plots'):
    os.makedirs('final_plots')

print(f"\nActuator dimensions: {ACTUATOR_WIDTH}mm x {ACTUATOR_HEIGHT}mm")
print("Creating plots and calculating strain...")

# Initialize list to collect all results for consolidated table
consolidated_results = []

# Process CSV files
for file in csv_files:
    if file == '.DS_Store':
        continue
        
    try:
        print(f"\nProcessing file: {file}")
        
        # Read the CSV file with chunksize for memory efficiency
        chunks = pd.read_csv(file, sep=';', decimal=',', chunksize=100000)
        
        # Initialize lists to store data
        times = []
        voltages = []
        displacements = []
        
        # Process each chunk
        for chunk in chunks:
            times.extend(chunk['Time (s)'])
            voltages.extend(chunk['Voltage (kV)'])
            displacements.extend(chunk['Laser (mm)'])
        
        # Convert to numpy arrays for signal processing
        times = np.array(times)
        voltages = np.array(voltages)
        displacements = np.array(displacements)
        
        # Flip displacement (multiply by -1)
        displacements = -displacements
        
        # Apply stronger Savitzky-Golay filter for smoother curves
        # Increase window length and adjust polynomial order
        window_length = min(5001, len(displacements) - 1 if len(displacements) % 2 == 0 else len(displacements) - 2)
        smooth_displacement = savgol_filter(displacements, window_length, 2)
        
        # Additional smoothing pass for even smoother curve
        smooth_displacement = savgol_filter(smooth_displacement, window_length, 2)
        
        # Calculate strain values (strain = displacement / original_length)
        strain_raw = (displacements / ACTUATOR_HEIGHT) * 100  # in percentage
        strain_smooth = (smooth_displacement / ACTUATOR_HEIGHT) * 100  # in percentage
        
        # Calculate y-axis limits based on this file's data
        file_min = min(np.min(displacements), np.min(smooth_displacement))
        file_max = max(np.max(displacements), np.max(smooth_displacement))
        
        # Add 10% padding to the range
        data_range = file_max - file_min
        if data_range > 0.001:  # Only if there's meaningful variation
            padding = data_range * 0.1
            y_min = file_min - padding
            y_max = file_max + padding
        else:
            # For flat signals, use a small fixed range around the value
            y_center = (file_max + file_min) / 2
            y_min = y_center - 0.1
            y_max = y_center + 0.1
        
        print(f"  Y-axis range for this file: {y_min:.3f} to {y_max:.3f} mm")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Extract weight and voltage for title
        weight_match = re.search(r'_(\d+)g', file)
        voltage_match = re.search(r'(\d+)V', file)
        weight = weight_match.group(1) if weight_match else "unknown"
        voltage = voltage_match.group(1) if voltage_match else "unknown"
        
        fig.suptitle(f'Raw Data - {weight}g load, {voltage}V\nFile: {file}', fontsize=12)
        
        # Plot 1: Voltage over time
        ax1.plot(times, voltages, 'b-', alpha=0.7, label='Voltage')
        ax1.set_xlabel('Time (s)', fontsize=14)
        ax1.set_ylabel('Voltage (kV)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        
        # Plot 2: Raw and smoothed displacement over time
        ax2.plot(times, displacements, 'r-', alpha=0.15, label='Raw Displacement', linewidth=1)
        ax2.plot(times, smooth_displacement, 'k-', alpha=1.0, label='Smoothed Displacement', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=14)
        ax2.set_ylabel('Displacement (mm)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        # Set y-axis limits based on this file's data
        ax2.set_ylim(y_min, y_max)
        
        # Add zero line if it's within the visible range
        if y_min <= 0 <= y_max:
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        # Extract just the filename without directory path
        base_filename = os.path.basename(file)
        clean_filename = base_filename.replace(' ', '_').replace('°', 'deg')
        plot_filename = f'final_plots/{clean_filename[:-4]}_plot.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"  Duration: {times[-1]:.2f} seconds")
        print(f"  Voltage range: {min(voltages):.3f} to {max(voltages):.3f} kV")
        print(f"  Raw displacement range: {min(displacements):.3f} to {max(displacements):.3f} mm")
        print(f"  Smoothed displacement range: {min(smooth_displacement):.3f} to {max(smooth_displacement):.3f} mm")
        print(f"  Raw strain range: {min(strain_raw):.3f}% to {max(strain_raw):.3f}%")
        print(f"  Smoothed strain range: {min(strain_smooth):.3f}% to {max(strain_smooth):.3f}%")
        print(f"  Peak strain (smoothed): {max(abs(strain_smooth)):.3f}%")
        
        # Calculate steady-state displacement (Δd)
        print("\n  Calculating steady-state displacement (Δd)...")
        periods_data = detect_on_off_periods(voltages, times, smooth_displacement, VOLTAGE_THRESHOLD)
        delta_d, periods_data, stats = detect_outliers_and_calculate_delta_d(periods_data)
        
        if delta_d is not None:
            print(f"  Steady-state displacement (Δd): {delta_d:.4f} mm")
            print(f"  Used {stats['good_on_periods']}/{stats['total_on_periods']} ON periods, "
                  f"{stats['good_off_periods']}/{stats['total_off_periods']} OFF periods")
            
            # Extract metadata from filename
            base_filename = os.path.basename(file)
            
            # Parse filename: e.g., "Contractile 2_60°_50g 2000V 11.11 14h56m57s.csv"
            # Sample: Contractile X
            sample_match = re.search(r'Contractile\s*(\d+\w*)', base_filename)
            sample = sample_match.group(1) if sample_match else "unknown"
            
            # Angle: XX°
            angle_match = re.search(r'(\d+)°', base_filename)
            angle = int(angle_match.group(1)) if angle_match else None
            
            # Load: XXXg
            load_match = re.search(r'(\d+)g', base_filename)
            load = int(load_match.group(1)) if load_match else None
            
            # Voltage: XXXXXv
            voltage_match = re.search(r'(\d+)V', base_filename)
            voltage_val = int(voltage_match.group(1)) if voltage_match else None
            
            # Determine if contracts or extends (+ displacement = contracts)
            behavior = "contracts" if delta_d > 0 else "extends"
            
            # Add to consolidated results
            consolidated_results.append({
                'sample': f"Contractile {sample}",
                'angle': angle,
                'load': load,
                'voltage': voltage_val,
                'displacement': round(delta_d, 4),
                'contracts or extends': behavior
            })
        else:
            print(f"  Warning: Could not calculate Δd - {stats.get('error', 'Unknown error')}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")

# Create consolidated results table - ONE FINAL TABLE WITH ALL SAMPLES
if consolidated_results:
    consolidated_df = pd.DataFrame(consolidated_results)
    consolidated_excel = 'final_plots/consolidated_results.xlsx'
    consolidated_df.to_excel(consolidated_excel, index=False, engine='openpyxl')
    print(f"\n{'='*70}")
    print(f"CONSOLIDATED RESULTS TABLE SAVED: {consolidated_excel}")
    print(f"Total samples processed: {len(consolidated_results)}")
    print(f"{'='*70}")
    print("\nFinal Consolidated Table:")
    print(consolidated_df.to_string(index=False))
    print(f"\n{'='*70}")
else:
    print("\nNo results to consolidate.")

print("\nAll plots saved in 'final_plots' directory.")