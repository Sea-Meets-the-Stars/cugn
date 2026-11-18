# Fuss with LaTeX

## Context

### High-level

You are helping me and Dan Rudnick to write a paper about the structure function of the velocity field to be submitted to the Journal of Physical Oceanography.  

### Writing style

See the PDFs in cugn/papers/Structure/Writing/previous/ for several of my previous publications so that you learn my writing style.

### Gutierrez-Villanueva manuscript

Read the Gutierrez-Villanueva manuscript `Gutierrez-Villanueva2025.pdf` in Writing/.  They describe in detail the QG model that we are using.

### Code

See all of the files in the cugn/papers/Structure/ and the calls to the main Repo and other dependencies. 

### Overleaf

The git repo for this paper may be found here:
`Projects/Overleaf/Energy_Cascade`

### Outline

Our outline is in the cugn/papers/Structure/Writing/ folder.  It is named outline.md.

## Fussing

1.  Presently, many of the section references use `\ref` give only a letter
instead of the section number and letter.  For example
`\ref{sec:structfunc}`.  Please examine the latex file and style file and 
let me know how to address this.
Put you comments and any questions in the Q&A/References section below.
Use Opus 5.5.  Log your work

## Q&A

### References

*Claude (2026-11-17), Fussing prompt 1.*

**Cause.** `ametsocV6.1.cls` (lines 417–419) numbers subsections with a bare
letter:

```latex
\renewcommand\thesection   {\@arabic\c@section}
\renewcommand\thesubsection{\@alph\c@subsection}
```

LaTeX builds the text of `\ref` as `\p@subsection\thesubsection`. The prefix
`\p@subsection` is empty by default, and the class never sets it. So
`\ref{sec:structfunc}` produces just "b", while the heading reads "b. Structure
functions" under "3. Observations and Methods". This is a class gap, not an error
in our `.tex`. The current `.aux` confirms it: every subsection label resolves to
a single letter (a–d), and the appendix ones to b–f.

**Fix (recommended).** Set the prefix in the preamble of `energy_cascade.tex`,
just before `\begin{document}`. This leaves the class file untouched, which is
safer for AMS submission:

```latex
% Cross-references to (sub)sections include the parent number, e.g. 2b (AMS style)
\makeatletter
\renewcommand\p@subsection{\thesection}
\renewcommand\p@subsubsection{\thesection\thesubsection}
\makeatother
```

This changes only what `\ref` prints; headings, numbering and hyperref anchors
stay as they are. I tested it on a scratch copy (not the Overleaf folder) with a
clean 2-pass compile and no errors. The labels now resolve to `sec:kh_2d_motivation` → 2a,
`sec:qg` → 2d, `sec:structfunc` → 3b, `sec:results_calypso` → 4a,
`sec:results_qg` → 4c, etc. The range "Sections~4a--4b" also works.

**Appendix wrinkle.** The paper calls the bare `\appendix` (the class's single-
appendix form), which sets `\thesection` = `A\arabic{section}`. The appendix uses
`\subsection` with no `\section`, so the section counter is 0. With the preamble
fix alone, the appendix references would print "A0b", "A0c", … That is wrong, so
a second line is needed right after `\appendixtitle{...}`:

```latex
\makeatletter\renewcommand\p@subsection{A}\makeatother  % -> Ab, Ac, ...
```

That gives "Appendix~Ab", etc. (tested). The headings themselves still read
"b. Sensitivity to separation binning". If you would rather have the reference
match the heading exactly, use `\renewcommand\p@subsection{}` instead. The text
would then read "Appendix~b", or better "section~b of the appendix".

**Related AMS style point (optional).** AMS journals write cross-references in
lowercase in running text: "section 2b", "appendix A", "Fig. 3". We currently use
"Section~\ref", "Appendix~\ref" and "Figure~\ref". The AMS copy editors will
change these anyway, but I can do a global swap now if you want.

**Questions**

1. Should I apply the preamble fix (plus the appendix line) to `energy_cascade.tex`?
   If so, do you want appendix references as **"Ab"** or as **"b"** (with "the
   appendix" wording)?
>
2. Should I convert "Section/Appendix/Figure~\ref" to AMS lowercase / "Fig."
   style at the same time?
>

## Logs

### 2026-11-17 (Fussing 1 — diagnosed letter-only section references)

Read `energy_cascade.tex` and `ametsocV6.1.cls`.
- **Cause:** the class sets `\thesubsection` to `\@alph` but never defines
  `\p@subsection`, so `\ref` to a subsection yields only the letter. Confirmed in
  the `.aux`: all subsection labels resolve to a–f.
- **Fix:** a preamble redefinition of `\p@subsection` / `\p@subsubsection`. A
  second issue is that the bare `\appendix` sets `\thesection` to "A0" (no
  `\section` in the appendix), so appendix references also need
  `\p@subsection{A}` or `{}` after `\appendixtitle`.
- **Test:** compiled both fixes on a copy in the scratchpad. 0 errors; the labels
  became 2a–2d, 3a–3c, 4a–4d and Ab–Af.
- Audited all 36 `~\ref{sec:...}` uses. "Appendix" is used only for appendix
  labels and "Section" only for body labels, so the wording is consistent.

The Overleaf `.tex` was **not modified**. The diagnosis, the fix, an optional AMS
lowercase-style note and two questions are in Q&A/References. Ran as Opus 5.5.
