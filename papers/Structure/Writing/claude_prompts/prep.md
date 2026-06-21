# Prep Claude for Writing this paper

## Context

### High-level

You are helping me and Dan Rudnick to write a paper about the structure function of the velocity field to be submitted to the Journal of Physical Oceanography.  

### Writing style

See the PDFs in cugn/papers/Structure/Writing/previous/ for several of my previous publications so that you learn my writing style.

### Code

See all of the files in the cugn/papers/Structure/ and the calls to the main Repo and other dependencies. 

### Overleaf

The git repo for this paper may be found here:
/home/xavier/Projects/Overleaf/Energy_Cascade

## Outline

1. Examine the Context. I have downloaded a PDF version of the Outline I had generated with Dan.  Please convert it into a Markdown file in the cugn/papers/Structure/Writing/ folder.  Call it outline.md.  Log your work in Logs

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-06-21 (Converted Outline PDF to Markdown)

Read `Outline.pdf` (2 pages) from `cugn/papers/Structure/Writing/` and converted it into `outline.md` in the same folder. The PDF is the joint outline developed with Dan Rudnick for the velocity structure-function paper (target: JPO).

Structure captured: Authorship, Journal, Text (Introduction, Theoretical underpinnings, Observations and Methods, Results, Discussion), and Figures. Preserved the PDF's annotations semantically:
- Green "[X]" markers (sections with figures/material ready) → bold **[X]**.
- Red text (open questions / TODO items, e.g. "linear or logarithmic?", "Calypso 2022", "Appendix") → italics.
- Strikethrough item (`<duL T^2>, <duL S^2>, etc.`) → ~~strikethrough~~.
- "Dan" callout for who writes the Data text → bold.

Rendered math expressions (`<du>`, `<duL>`, `<du^3>`, `N_eddies`, etc.) as LaTeX so they read cleanly. Key scientific content noted: Karman-Howarth theory (Rick Salmon), 2D being critical for the experiment, QG analysis, three field experiments (2 Calypso + 1 ARCTERX), and the central message that humans cannot measure the sign of the energy cascade.