# Drift in QG — Claude Guide

## Project Overview

This project investigates Lagrangian drifter trajectories within a 2D quasigeostrophic (QG) ocean model. The goal is to deploy virtual drifters in the QG flow field, compute structure functions from their separations, and compare with observational structure functions from Spray gliders (CUGN data).

## Communication Style
- Be direct and concise. No sycophantic preamble.
- When uncertain, say so explicitly.
- Be critical of prompts; do not simply aim to please.

## Code Conventions
- **Julia** is the primary language. Use MeshArrays.jl and IndividualDisplacements.jl for grid handling and drifter advection.
- **Python** may be used for plotting and post-processing. Conda environment: `ocean14`.
- Separate analysis scripts from figure scripts so figures can be regenerated without rerunning analysis.
- Reuse existing utilities in `papers/Structure/Analysis/py/qg_utils.py` where applicable.
- Include inline comments in the code to explain what is happening.
- Put import statements at the top of the file.

## Directory Structure

```
papers/Structure/Analysis/QG/
├── jl/              # Julia analysis scripts
├── py/              # Python helper scripts
├── Claude.md        # This file — project-specific Claude guidance
└── drift_in_qg.md  # Prompt log and project organization
```

## Key Data

- **QG model output**: `$OS_DATA/QG/QGModelOutput20years.nc` — 20 years of 2D QG simulation
- **Analysis outputs**: `papers/Structure/Analysis/Outputs/`
- **Existing QG analysis scripts**: `papers/Structure/Analysis/py/qg_utils.py`, `qg_100km.py`, `qg_uL_SF.py`

## Julia Packages

- [MeshArrays.jl](https://github.com/JuliaClimate/MeshArrays.jl) — grid/mesh data structures.  A fork of the package is found here: /home/xavier/Projects/Oceanography/julia/MeshArrays.jl
- [Drifers.jl](https://github.com/JuliaClimate/Drifers.jl) — Lagrangian particle tracking.  A fork of the package is found here: /home/xavier/Projects/Oceanography/julia/Drifers.jl


## LaTeX / Overleaf Conventions
- Overleaf project: `/home/xavier/Projects/overleaf/Structure_function`
- Figures go in the Overleaf project folder as PNG files with informative filenames.
- Author: `JXP & Claude [model version]`.
- Documents show creation date and last-edit date at the top.
- Embed figures within the text near their description.
- Maintain a chronological change log documenting requests and changes implemented.
- You may push to git as you work. The access token is in `.bashrc` with the name OVERLEAF.

## Bash Commands
- Safe bash commands may be run without prompting.
- Multiple agents may be used to parallelize work.
