# VPSC-GUI

**VPSC-GUI** is a Python graphical pre- and post-processing environment for simulations performed with the Visco-Plastic Self-Consistent (VPSC) code. The program keeps the original VPSC8 Fortran solver unchanged and provides a structured graphical workflow for editing VPSC input files, launching a user-supplied external solver and visualizing standard VPSC output files.

The software is designed for researchers working on crystal plasticity, texture evolution, polycrystalline plasticity, plastic anisotropy, Lankford coefficients, yield loci and deformation-mode activity in metallic materials.

## Main features

- Project configuration for VPSC case folders, control files, external executables and output directories.
- Structured editing of `VPSC8.IN`, single-crystal `.sx` files, texture files and process/boundary-condition files.
- Pole figure and inverse pole figure visualization for cubic and HCP textures.
- Boundary-condition visualization for velocity-gradient and Cauchy-stress constraints.
- External VPSC8 solver execution from a reproducible run directory.
- Direct post-processing of common VPSC outputs, including:
  - `STR_STR.OUT` for stress-strain curves,
  - `RUN_LOG.OUT` for solver logs,
  - `TEX_PH*.OUT` for texture evolution,
  - `PCYS.OUT` for polycrystal yield loci,
  - `LANKFORD.OUT` for Lankford coefficients and directional Young's moduli,
  - `ACT_PH*.OUT` for relative slip/twin activity.
- Publication-oriented plotting with Matplotlib.

All representative figures in the associated SoftwareX manuscript were generated directly within VPSC-GUI.

## Important note on the VPSC executable

This repository does **not** redistribute the original VPSC/VPSC8 executable. VPSC-GUI is a graphical workflow layer around an external solver. Users must provide their own properly licensed VPSC executable and select it in the **Project** panel before running simulations.

For this reason, files such as `vpsc8.exe`, `VPSC8.EXE` or other third-party executables should not be committed to this repository.

## Repository structure

A typical repository layout is:

```text
VPSC-GUI/
├── src/
│   └── vpsc_gui/
│       ├── __init__.py
│       ├── __main__.py        # optional launcher
│       └── app.py             # main GUI and scientific utilities
├── examples/
│   ├── FCC_rolling/
│   └── HCP_compression/
├── tests/
├── README.md
├── LICENSE
├── CITATION.cff
├── MANIFEST.in
└── pyproject.toml
```

The example folders should contain only input files and lightweight demonstration data. Generated run folders and solver outputs are ignored by `.gitignore`.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ZhiyangNi/VPSC-GUI.git
cd VPSC-GUI
python -m pip install -e .
```

The required Python packages are:

```bash
python -m pip install numpy matplotlib
```

`scipy` is optional and is used only for smoother density-map filtering when available:

```bash
python -m pip install scipy
```

Tkinter is required to launch the graphical interface. It is included in many Python distributions. On Debian/Ubuntu, it may need to be installed separately:

```bash
sudo apt install python3-tk
```

## Running the application

After installation:

```bash
vpsc-gui
```

or, from the repository root:

```bash
python src/vpsc_gui/app.py
```

The self-test can be run without launching the graphical interface:

```bash
python src/vpsc_gui/app.py --self-test
```

or with a specified working directory:

```bash
python src/vpsc_gui/app.py --self-test examples
```

## Basic workflow

1. Open the **Project** panel and choose:
   - base case folder,
   - `VPSC8.IN`,
   - external VPSC8 executable,
   - output root directory.
2. Use the **VPSC8.IN**, **Single Crystal**, **Texture** and **Process / BC** panels to inspect and edit input files.
3. Prepare and run the case in the **Run** panel.
4. Use the **Results** panel to draw solver logs, stress-strain curves, pole figures, yield loci, Lankford coefficients, Young's moduli and relative activity curves.
5. Export the generated figures for reports or publications.

## Examples

The repository contains two representative examples.

### `examples/FCC_rolling`

This example demonstrates a typical FCC rolling simulation and post-processing workflow. It is used to show solver logs, stress-strain response, pole figure visualization, polycrystal yield locus, Lankford coefficient and directional Young's modulus.

### `examples/HCP_compression`

This example demonstrates compression of an HCP magnesium aggregate. It is used to show initial and deformed (0002) pole figures, the macroscopic stress-strain curve and the relative activity of basal slip, prismatic slip, pyramidal slip and tensile twinning.

## Citation

If you use VPSC-GUI in your research, please cite the associated SoftwareX article and the software release. A machine-readable citation file is provided as `CITATION.cff`.

```bibtex
@software{vpsc_gui_2026,
  author  = {Ni, Zhiyang and Guo, Min and Chen, Zhanghua and Dong, Jianxin and Jiang, He},
  title   = {VPSC-GUI: A Python graphical pre- and post-processing environment for VPSC8},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/ZhiyangNi/VPSC-GUI}
}
```

## License

VPSC-GUI is released under the BSD 3-Clause License. See `LICENSE` for details.

The original VPSC/VPSC8 solver is not included in this repository and is not covered by this license.

## Contact

For questions related to this software, please contact:

**He Jiang**  
School of Materials Science and Engineering  
University of Science and Technology Beijing  
Email: jianghe17@sina.cn
