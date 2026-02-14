# Figures for Papers
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Chen-blue)](https://www.linkedin.com/in/chenliu1996/)
[![Twitter Follow](https://img.shields.io/twitter/follow/Chen.svg?style=social)](https://x.com/ChenLiu_1996)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Chen-4a86cf?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=3rDjnykAAAAJ&sortby=pubdate)

This is a centralized repository of my own **Python scripts for high-quality figures**.

I am [Chen Liu](https://chenliu-1996.github.io/), a Computer Science PhD Candidate at Yale University.


### Bar plots for quantitative comparison
<img src="figure_ImmunoStruct/figures/bars_comparison_IEDB.png" width="800">

<img src="figure_CellSpliceNet/figures/comparison.png" width="800">

### Bar plots for composition breakdown
<img src="figure_brainteaser/figures/brute_force.png" width="800">

### Trend plots
<img src="figure_ophthal_review/figures/trend_by_month.png" width="800">

### Heat maps
<img src="figure_RNAGenScape/figures/results_comparison_optimization.png" width="800">

### 3D spheres
<img src="figure_Dispersion/figures/illustration.png" width="800">

### Miscellaneous: figures not made end-to-end in Python
These figures were made partially in Python. I included them to acknowledge the time and efforts I spent on them.

<img src="assets/ImmunoStruct_schematic.png" width="400"><img src="assets/ImmunoStruct_contrastive.png" width="400">
<br><img src="assets/ImmunoStruct_results_IEDB.png" width="400"><img src="assets/ImmunoStruct_results_CEDAR.png" width="400">
<br><img src="assets/RNAGenScape_schematic.png" width="400"><img src="assets/Dispersion_motivation.png" width="400">
<br><img src="assets/Dispersion_observation.png" width="400"><img src="assets/Dispersion_observation_distillation.png" width="400">

<details>
<summary><strong>How to use the Scientific Figure Pro skill (click to expand)</strong></summary>

<br>

The repository includes an LLM skill guide plus reusable helper scripts:

- Skill guide: `skills/scientific-figure-pro/SKILL.md`
- Style rationale: `DESIGN_THEORY.md`
- Helper implementation: `skills/scientific-figure-pro/scripts/scientific_figure_pro.py`

### Simple AI workflow

1. Open this repository in Cursor.
2. Ask the AI to create or update a plotting script in your target folder (for example `figure_PROJECT_NAME/`).
3. In your prompt, explicitly ask it to follow `skills/scientific-figure-pro/SKILL.md` and `DESIGN_THEORY.md`.
4. Run the generated script and check the exported figure.

### Prompt template (copy/paste)

```text
Create a publication-quality figure script at <target_path>.
Use the Scientific Figure Pro skill conventions from:
- skills/scientific-figure-pro/SKILL.md
- DESIGN_THEORY.md

Load and use `skills/scientific-figure-pro/scripts/scientific_figure_pro.py` (apply_publication_style, make_* helpers, finalize_figure).
Input data: <describe your data or paste arrays>.
Output files: <name>.png and <name>.pdf.
Keep the style consistent with this repository.
```

</details>
