# SoftwareX paper outline for VPSC-GUI

## Proposed title

VPSC-GUI: A Python graphical pre- and post-processing environment for the Visco-Plastic Self-Consistent code

## Keywords

Visco-Plastic Self-Consistent model; crystal plasticity; VPSC8; texture visualization; finite-element velocity gradient; Python GUI

## Article structure

1. Motivation and significance
2. Software description: VPSC input generation and editing
3. Software description: texture visualization and post-processing
4. Illustrative examples
5. Conclusions

## Suggested figures

### Figure 1. VPSC-GUI workflow
Show the complete workflow from VPSC input files to GUI editors, external VPSC8 solver execution and post-processing outputs.

### Figure 2. Structured input and boundary-condition editing
Show VPSC8.IN editing, single-crystal `.sx` editing, process/boundary-condition construction and boundary-condition visualisation.

### Figure 3. Texture visualization
Show FCC pole figures, FCC inverse pole figures, HCP `(0002)` and `(10-10)` pole figures, and HCP inverse pole figures.

### Figure 4. Post-processing examples
Show stress-strain response, slip activity, Lankford coefficient and polycrystal yield surface.

## Code metadata draft

| Item | Description |
|---|---|
| Current code version | V1.0.0 |
| Permanent link to code | To be added after GitHub release |
| Legal code license | BSD-3-Clause |
| Code versioning system used | git |
| Software code language | Python |
| Compilation requirements | Python packages listed in `pyproject.toml` |
| Link to developer documentation | To be added after GitHub release |
| Support email for questions | To be added |

## Notes for submission

- Freeze the software release before manuscript submission.
- Create a public GitHub repository and a versioned release tag.
- Archive the release using Zenodo or an equivalent service.
- Replace placeholder author, repository and DOI fields before submission.
