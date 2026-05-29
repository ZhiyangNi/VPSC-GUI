# VPSC-GUI

VPSC-GUI is a Python graphical pre- and post-processing environment for simulations performed with the Visco-Plastic Self-Consistent (VPSC) code. The program keeps the original VPSC8 Fortran solver unchanged and provides a structured graphical workflow for editing VPSC input files, launching a user-supplied external solver and visualizing standard VPSC output files.

The software is designed for researchers working on crystal plasticity, texture evolution, polycrystalline plasticity, plastic anisotropy, Lankford coefficients, yield loci and deformation-mode activity in metallic materials.

All representative figures in the associated SoftwareX manuscript were generated directly within VPSC-GUI.

## Main features

* Project configuration for VPSC case folders, control files, external executables and output directories.
* Structured editing of `VPSC8.IN`, single-crystal `.sx` files, texture files and process/boundary-condition files.
* Pole figure and inverse pole figure visualization for cubic and HCP textures.
* Boundary-condition visualization for velocity-gradient and Cauchy-stress constraints.
* External VPSC8 solver execution from a reproducible run directory.
* Direct post-processing of common VPSC output files, including:

  * `STR_STR.OUT` for stress-strain curves,
  * `RUN_LOG.OUT` for solver logs,
  * `TEX_PH*.OUT` for texture evolution,
  * `PCYS.OUT` for polycrystal yield loci,
  * `LANKFORD.OUT` for Lankford coefficients and directional Young's moduli,
  * `ACT_PH*.OUT` for relative slip/twin activity.
* Publication-oriented plotting with Matplotlib.

## Important note on the VPSC solver

This repository does not include or redistribute the original VPSC executable.

Users should obtain, compile or provide their own properly licensed VPSC8 executable. VPSC-GUI only provides the Python graphical workflow layer for input preparation, solver execution and post-processing.

## Installation

Clone the repository:

```bash
git clone https://github.com/ZhiyangNi/VPSC-GUI.git
cd VPSC-GUI
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Running the app

The recommended launch command is:

```bash
python -m vpsc_gui
```

If the console script is generated successfully during installation, the app can also be launched with:

```bash
vpsc-gui
```

## Basic workflow

1. Open VPSC-GUI.
2. Go to the **Project** panel.
3. Select the base directory of a VPSC case.
4. Select the `VPSC8.IN` file.
5. Select the local VPSC8 executable.
6. Apply the project paths.
7. Inspect or edit the input files using:

   * **VPSC8.IN**,
   * **Single Crystal**,
   * **Texture**,
   * **Process / BC**.
8. Use the **Run** panel to prepare a run directory and launch the external solver.
9. Use the **Results** panel to visualize and export output figures.

## Examples

The repository contains two representative examples:

```text
examples/
├── FCC_rolling/
└── HCP_compression/
```

### FCC_rolling

This example demonstrates an FCC rolling case and can be used to test:

* VPSC input-file loading,
* texture preview,
* boundary-condition visualization,
* solver execution,
* stress-strain plotting,
* pole figure visualization,
* polycrystal yield locus plotting,
* Lankford coefficient plotting,
* directional Young's modulus plotting.

### HCP_compression

This example demonstrates an HCP magnesium compression case and can be used to test:

* HCP texture visualization,
* initial and deformed `(0002)` pole figures,
* compressive stress-strain response,
* relative activity of deformation modes such as basal slip, prismatic slip, pyramidal slip and tensile twinning.

## Repository structure

```text
VPSC-GUI/
├── docs/
├── examples/
│   ├── FCC_rolling/
│   └── HCP_compression/
├── src/
│   └── vpsc_gui/
│       ├── __init__.py
│       ├── __main__.py
│       └── app.py
├── tests/
├── .gitignore
├── CITATION.cff
├── LICENSE
├── MANIFEST.in
├── README.md
└── pyproject.toml
```

## Dependencies

VPSC-GUI requires Python 3.9 or later.

The main Python dependencies are:

```text
numpy
matplotlib
```

The graphical interface is based on Tkinter, which is included with most standard Python installations.

## Testing installation

After installation, run:

```bash
python -m vpsc_gui
```

If the graphical interface opens correctly, the installation is successful.

## License

VPSC-GUI is distributed under the BSD-3-Clause License.

The original VPSC solver is not part of this repository and is not redistributed under this license.

## Citation

If you use VPSC-GUI in academic work, please cite the associated SoftwareX article and the repository.

A citation file is provided in:

```text
CITATION.cff
```

## Contact

For questions, suggestions or bug reports, please use the GitHub issue tracker:

```text
https://github.com/ZhiyangNi/VPSC-GUI/issues
```
