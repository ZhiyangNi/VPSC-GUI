# VPSC-GUI

**VPSC-GUI** is a Python graphical pre- and post-processing environment for the
Visco-Plastic Self-Consistent (VPSC8) code. It provides a structured workflow for
preparing VPSC input files, visualising process and boundary conditions, running a
user-supplied external VPSC8 executable, and post-processing standard VPSC output
files.

The program is an optional front end around the original VPSC8 Fortran solver. It
does **not** modify or redistribute the VPSC8 solver or executable.

## Current version

```text
v1.0.1
```

This revision adds phase-aware support for native FCC/BCC two-phase VPSC cases and
improves the publication layout of pole-figure and inverse-pole-figure plots.

## Main features

- Structured preview and editing of `VPSC8.IN`.
- Phase-aware dashboard for single-phase and multi-phase VPSC cases.
- Explicit phase selection for phase-specific single-crystal `.sx` files.
- Explicit phase selection for phase-specific texture files.
- Single-crystal `.sx` file preview and editing for FCC, BCC/cubic and HCP materials.
- Texture preview, pole-figure and inverse-pole-figure visualization.
- Boundary-condition and process-file editor with common loading presets.
- Visualization of velocity-gradient matrices, symmetric rate tensors, Cauchy-stress constraints and deformation sketches.
- Import and conversion of VPSC7-style, VPSC8-style and generic finite-element velocity-gradient histories.
- Fortran-compatible run directory generation with dependency checks for all phase-specific input files.
- Optional-file checking for grain-shape and diffraction files referenced in `VPSC8.IN`.
- Phase-resolved post-processing of common VPSC output files, including `ACT_PH*.OUT` and `TEX_PH*.OUT`.
- Direct post-processing of `STR_STR.OUT`, `LANKFORD.OUT`, `PCYS.OUT`, `RUN_LOG.OUT` and related VPSC output files.
- High-resolution export of figures in PNG, TIFF, JPEG, PDF and SVG formats.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/ZhiyangNi/VPSC-GUI.git
cd VPSC-GUI
python -m pip install -e .
```

For testing utilities:

```bash
python -m pip install -e .[test]
```

## Running the GUI

After installation, run:

```bash
vpsc-gui
```

or:

```bash
python -m vpsc_gui
```

The GUI can also be started directly from the source tree:

```bash
PYTHONPATH=src python -m vpsc_gui
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="src"
python -m vpsc_gui
```

## Self-test

A non-GUI smoke test is provided to check the main parsers and texture rendering routines:

```bash
PYTHONPATH=src python -m vpsc_gui --self-test examples/FCC_rolling
PYTHONPATH=src python -m vpsc_gui --self-test examples/BCC_FCC_Tension
```

The self-test parses the example VPSC input files and generates several test pole-figure and inverse-pole-figure images in the selected example directory.

## Example data

The repository includes three representative examples:

```text
examples/
├── FCC_rolling/
├── HCP_compression/
└── BCC_FCC_Tension/
```

### FCC_rolling

A single-phase FCC rolling example for testing input-file loading, process visualization, solver staging, stress-strain plotting, pole-figure visualization, polycrystal yield locus plotting and Lankford-coefficient post-processing.

### HCP_compression

A single-phase HCP magnesium compression example for testing HCP pole figures, compressive stress-strain response and relative slip/twin activity plotting.

### BCC_FCC_Tension

A native-style two-phase VPSC8 example for FCC/BCC phase-aware workflows. It uses:

```text
FCC.sx
BCC.sx
FCC_texture.tex
BCC_texture.tex
tension
vpsc8.in
```

The example is intended to test phase-aware parsing of `vpsc8.in`, explicit FCC/BCC single-crystal selection, phase-specific texture preview, dependency checking during run preparation, and result selection for native two-phase outputs such as:

```text
ACT_PH1.OUT
ACT_PH2.OUT
TEX_PH1.OUT
TEX_PH2.OUT
```

## Multi-phase VPSC workflows

VPSC-GUI v1.0.1 includes phase-aware support for VPSC cases with more than one phase. After `VPSC8.IN` is loaded, the Dashboard reports all detected phases, volume fractions and phase-specific files. The Single Crystal and Texture pages provide phase selectors for the corresponding `.sx` and texture files. During run preparation, the app copies all referenced phase-specific files and writes `app_dependency_report.txt` to the run directory. After a solver run, the Results page can select phase-resolved `ACT_PHn.OUT` and `TEX_PHn.OUT` files for activity and texture plotting.

Optional grain-shape and diffraction files referenced by `VPSC8.IN` are checked as dependencies. VPSC-GUI does not interpret these optional files physically; it reports missing files before execution so that users can correct the case directory or the input file.

## Pole-figure and inverse-pole-figure visualization

VPSC-GUI uses Bunge Euler angles consistently with the VPSC sample-axis convention. The implemented convention is:

```text
Pole figure:          v_sample  = g^T · v_crystal
Inverse pole figure:  v_crystal = g   · v_sample
```

Here `g` is the sample-to-crystal orientation matrix constructed from the Bunge Euler angles. Pole figures therefore rotate crystallographic pole normals into the sample frame, whereas inverse pole figures rotate a sample direction such as RD, TD or ND into the crystal frame. The default density mode uses an equal-area projection with mean-density normalization, so random textures produce approximately unit relative density and different phases can be compared on a consistent visualization basis.

The colorbar is intentionally left without a text title in the embedded GUI to avoid label overlap in multi-panel figures. The values should be interpreted as relative pole-density intensities for visualization and comparison rather than as an experimentally calibrated diffraction intensity scale.

## External VPSC8 solver

VPSC-GUI is a front-end and post-processing environment. To run actual VPSC simulations, users should provide their own VPSC8 executable obtained from the official VPSC distribution. The executable path can be selected in the GUI.

## Repository layout

```text
VPSC-GUI/
  README.md
  pyproject.toml
  CITATION.cff
  LICENSE
  examples/
    FCC_rolling/
    HCP_compression/
    BCC_FCC_Tension/
  tests/
  docs/
  src/
    vpsc_gui/
      app.py
      __init__.py
      __main__.py
```

## License

VPSC-GUI is released under the BSD 3-Clause License. The external VPSC8 Fortran solver is not redistributed in this repository and should be cited and licensed according to its original distribution.

## Citation

If you use VPSC-GUI, please cite the archived software release and the associated SoftwareX article once available. A citation file is provided in `CITATION.cff`.
