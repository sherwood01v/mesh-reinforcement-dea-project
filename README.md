# Mesh Reinforcement for Soft Artificial Muscle

Semester project on fiber-reinforced dielectric elastomer actuators (DEAs). This repository contains the mathematical model and measurement analysis tools.

## Files

### Code

**`refined_mathematical_model.py`**
- Mathematical model for fiber-reinforced DEA behavior
- Based on Yeoh hyperelastic model with anisotropic fiber reinforcement
- Simulates actuation response under applied voltage and mechanical loads
- Accounts for fiber reorientation during deformation

**`raw_measurement_visualisation.py`**
- Processes experimental measurement data from CSV files
- Generates plots showing voltage input and actuator displacement response
- Applies Savitzky-Golay filtering for noise reduction
- Calculates strain values based on actuator geometry (30mm × 70mm)
- Detects ON/OFF periods and extracts steady-state behavior

### Videos

**`demo1_60_degree_0g_load_8kV.mp4`** and **`demo2_60_degree_0g_load_8kV.mp4.mp4`**
- Demonstration of actuator contraction at 60° fiber angle with 8kV applied voltage

## Requirements

```bash
pip install numpy matplotlib scipy pandas
```

## Usage

### Running the Mathematical Model

```bash
python refined_mathematical_model.py
```

This generates plots showing:
- Actuation strain vs. voltage at different loads
- Effect of fiber angle on performance
- Influence of fiber modulus
- Actuation response under varying conditions

### Processing Measurement Data

```bash
python raw_measurement_visualisation.py
```

The script processes CSV files in the current directory and generates:
- Voltage vs. time plots
- Displacement vs. time plots (raw and smoothed)
- Strain calculations
- Organized output by fiber angle

**Expected CSV filename format:**
```
Name_Angle°_Number_Weight Voltage Date Time.csv
```

Example:
```
Contractile 2_60°_0g 7000V 11.11 15h04m04s.csv
```

**Expected CSV file structure:**
- Separator: semicolon (`;`)
- Decimal separator: comma (`,`)
- Required columns:
  - `Time (s)` - timestamp in seconds
  - `Voltage (kV)` - applied voltage in kilovolts
  - `Laser (mm)` - displacement measurement in millimeters

## Parameters

Actuator dimensions (can be modified in both scripts):
- Width: 30 mm
- Height: 70 mm
- Active layer thickness: 100 μm
- Passive layer thickness: 20 μm per side

Material properties (Yeoh model coefficients):
- C10 = 281200 Pa
- C20 = -8087 Pa
- C30 = 976.6 Pa
- Relative permittivity: 1.5

## Output

Both scripts save results as PNG files with 300 dpi resolution. The visualization script organizes outputs by fiber angle in separate folders.

