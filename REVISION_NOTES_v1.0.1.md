# VPSC-GUI v1.0.1 revision notes

This revision was prepared in response to reviewer comments on multi-phase VPSC
support and file-dependency handling.

## Software changes

1. Added phase-aware Single Crystal editing.  The Single Crystal page now provides
   a phase selector populated from `vpsc8.in`, so the user can explicitly inspect
   or edit the `.sx` file of Phase 1, Phase 2, etc.
2. Added phase-aware Texture preview.  The Texture page now provides a phase
   selector populated from the phase-specific texture files in `vpsc8.in`.
3. Added phase-aware Results selection for native VPSC output names.  The Results
   page discovers `ACT_PHn.OUT` and `TEX_PHn.OUT` files and provides a
   `Result phase (ACT_PHn / TEX_PHn)` selector for phase-resolved activity and
   pole-figure plotting.  Numeric matching also accepts zero-padded names such as
   `TEX_PH02.OUT`.
4. Strengthened run-directory preparation.  VPSC-GUI now collects and stages all
   phase-dependent `.sx`, texture and process files referenced by `vpsc8.in`.
   Optional grain-shape and diffraction files are checked and reported without
   being interpreted physically by the GUI.
5. Added `app_dependency_report.txt` to each prepared run directory, listing all
   staged dependencies and any missing optional or required files.
6. Added a native-style two-phase example, `examples/BCC_FCC_Tension`, based on the
   VPSC8 two-phase layout.  The example uses `FCC.sx`, `BCC.sx`, a process file
   named `tension`, and documents expected outputs such as `ACT_PH1.OUT`,
   `ACT_PH2.OUT`, `TEX_PH1.OUT` and `TEX_PH2.OUT`.

## Documentation changes

- Updated README example list and workflow description.
- Clarified that the VPSC8 executable is not redistributed.
- Documented expected two-phase output naming conventions.

## Validation

- `python -m py_compile src/vpsc_gui/app.py`
- `PYTHONPATH=src pytest -q`
- `PYTHONPATH=src python -m vpsc_gui --self-test examples/BCC_FCC_Tension`

## Final PF/IPF visualization cleanup for reviewer response

- Refined the PF/IPF layout so Miller-index labels such as `(100)`, `(110)` and `(111)` are drawn explicitly above the TD mark and centered on the pole-figure vertical axis.
- Removed the `MRD` colorbar title from all pole-figure and inverse-pole-figure color bars to prevent the label from overlapping the colorbar/tick region in embedded GUI canvases.
- Increased multi-panel pole-figure canvas width, vertical headroom and colorbar padding to prevent overlaps between RD/TD labels, Miller labels, colorbar ticks and the circular rim.
- Audited the PF/IPF theory implementation: Bunge Euler angles are treated as the sample-to-crystal orientation matrix `g`; pole figures map crystal pole normals to sample axes through `g.T`; inverse pole figures map sample directions to crystal axes through `g`; projections use a common active projection for points, density grids and boundaries, with equal-area/Schmidt projection as the default for density maps.
- The density engine is normalized by the mean density over the valid projected domain, so a random texture is close to one and contour maps remain comparable across phases and pole families. The colorbar is intentionally left unlabeled in the GUI; the manuscript can describe the values as relative pole-density intensities.
- Final publication-layout pass: raised Miller-index labels further above the top axis label, slightly lowered the top axis label itself, and increased the pole-figure top extent so labels never touch in FCC/BCC multi-panel figures prepared for the manuscript.
