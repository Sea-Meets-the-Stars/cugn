# Glider Analysis

**IMPORTANT:** At the start of every conversation involving this project, you MUST read the local `Claude.md` file in this directory (`papers/Structure/Analysis/QG/Claude.md`) and follow its instructions. Do NOT rely solely on the top-level CUGN `CLAUDE.md` — the local one contains project-specific guidance, conventions, and context.

## Context

Now that we have a working code to examine the QG model with gliders, we need to generate code to make the glider measurements and analyze the outputs.  We expect to analyze the outputs for several different start times and locations within the domain. 

We need a Python module to guide the process.  Please name it py/glider_analysis.py.

Refer to the gliders_in_qg.md file for the existing code for measuring velocities from the QG model with gliders.

The module should be similar to the small_box_drifters.py module.

## Claude Code

- You should be critical of any prompts and not simply aim to please.
- Julia is the primary language for this project; Python may be used for plotting/analysis as needed.
- You are allowed to run safe bash commands without prompting.
- You are welcome to use multiple agents to help you with the task.
- When possible, reuse existing code and modules rather than writing new code.

# Prompts

## Plan

1. Read this document and develop a plan for the analysis.  Write it down in Overleaf.  Append it to the claude_gliders_in_qg_plan.tex file. Do not execute any code yet.
2. Turn the plan into a set of requirements for the code and put them in the Requirements section above.  Answers to the open questions in the planning doc are given in the Requirements section above.

## Code

1. Reread this doc. Generate the code to satisfy the Requirements.  Place the Python code in a py/qg_gliders.py module. Place the Julia code in a jl/qg_gliders.jl module.

## Tests

1. Reread this doc. Create the tests described in the Testing section above.
2. Examining the test_glider_velocity_field.png figure, the velocities of the gliders do not appear to match the underlying velocity field.  Please investigate. 