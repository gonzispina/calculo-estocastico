# Score-based generative modeling through stochastic differential equations

This repository contains a course monograph and an English presentation plan on the mathematical foundations of score-based diffusion models.

The project follows one reproducible one-dimensional example:

1. start from an unknown shifted-Gamma distribution;
2. diffuse it with an Ornstein-Uhlenbeck SDE until it approaches a Gaussian prior;
3. learn the time-dependent score with denoising score matching;
4. integrate the reverse-time SDE and the probability-flow ODE;
5. compare generated and reference samples with a Kolmogorov-Smirnov test.

The site version presents the poster as a sequence of short sections rather than as a single dense image. The complete poster and the monograph are available as downloadable PDFs.

## Contents

- `main.tex`: original Spanish monograph.
- `main_en.tex`: working scaffold for the English translation; the Spanish source remains authoritative while the chapter-by-chapter translation is checked.
- `poster.tex`: original Spanish poster.
- `poster_en.tex`: English poster source.
- `score_matching_demo.py`, `ou_trajectories.py`, `diffusion_3d.py`: numerical demonstrations and figure generation.
- `references.bib`: bibliography.
- `fig*.png`, `*_trajectories.png`, `score_matching_demo.png`: figures used by the documents.
- `papers/`: source papers used in the exposition; not required to run the demo.

## Build

The documents use standard LaTeX plus `biber`:

```bash
mkdir -p build
latexmk -pdf -use-biber -outdir=build main_en.tex
latexmk -pdf -outdir=build poster_en.tex
```

The figures can be regenerated with Python from the repository root. The scripts are intended as transparent demonstrations, not as a full image-generation implementation.

## Scope

This is a mathematical and computational exposition of the forward/reverse diffusion framework. The numerical example is deliberately one-dimensional: it makes the role of the score and the reverse dynamics inspectable without claiming to reproduce a modern image model.

## Portfolio presentation

The accompanying project page is published at [gonzalospina.com/projects/score-based-diffusions.html](https://gonzalospina.com/projects/score-based-diffusions.html). It develops the poster block by block and links back here for the source, code, and complete PDFs. The site currently links the original Spanish PDFs until the English PDFs are compiled and checked.
