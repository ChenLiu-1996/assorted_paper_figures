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

<br>

## LLM skill integration (credit to my friend [Shan Chen](https://shanchen.dev/))

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


## Related Papers
<details>
<summary>ImmunoStruct</summary>

[![nature](https://img.shields.io/badge/nature-machine_intelligence-white)](https://www.nature.com/articles/s42256-025-01163-y)
[![PDF](https://img.shields.io/badge/PDF-DADBDD)](https://www.nature.com/articles/s42256-025-01163-y.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/KrishnaswamyLab/ImmunoStruct.svg?style=social\&label=Stars)](https://github.com/KrishnaswamyLab/ImmunoStruct)
```bibtex
@article{givechian2025immunostruct,
  title={ImmunoStruct enables multimodal deep learning for immunogenicity prediction},
  author={Givechian, Kevin Bijan and Rocha, Jo{\~a}o Felipe and Liu, Chen and Yang, Edward and Tyagi, Sidharth and Greene, Kerrie and Ying, Rex and Caron, Etienne and Iwasaki, Akiko and Krishnaswamy, Smita},
  journal={Nature Machine Intelligence},
  pages={1--14},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

</details>
<details>
<summary>Dispersion</summary>

[![arXiv](https://img.shields.io/badge/arXiv-Dispersion-firebrick)](https://arxiv.org/abs/2602.00217) 
[![PDF](https://img.shields.io/badge/PDF-silver)](https://arxiv.org/pdf/2602.00217)
```bibtex
@article{liu2026dispersion,
  title={Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models},
  author={Liu, Chen and Sun, Xingzhi and Xiao, Xi and Van Tassel, Alexandre and Xu, Ke and Reimann, Kristof and Liao, Danqi and Gerstein, Mark and Wang, Tianyang and Wang, Xiao and others},
  journal={arXiv preprint arXiv:2602.00217},
  year={2026}
}
```

</details>
<details>
<summary>RNAGenScape</summary>

[![arXiv](https://img.shields.io/badge/arXiv-RNAGenScape-firebrick)](https://arxiv.org/abs/2510.24736) 
[![PDF](https://img.shields.io/badge/PDF-silver)](https://arxiv.org/pdf/2510.24736)
```bibtex
@article{liao2025rnagenscape,
  title={RNAGenScape: Property-Guided, Optimized Generation of mRNA Sequences with Manifold Langevin Dynamics},
  author={Liao, Danqi and Liu, Chen and Sun, Xingzhi and Tang, Di{\'e} and Wang, Haochen and Youlten, Scott and Gopinath, Srikar Krishna and Lee, Haejeong and Strayer, Ethan C and Giraldez, Antonio J and others},
  journal={arXiv preprint arXiv:2510.24736},
  year={2025}
}
```

</details>
<details>
<summary>brainteaser</summary>

[![OpenReview](https://img.shields.io/badge/OpenReview-eeeeee)](https://openreview.net/forum?id=3oQDkmW72a)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS_2025-purple)](https://openreview.net/pdf?id=3oQDkmW72a)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace_Dataset-orange)](https://huggingface.co/datasets/ChenLiu1996/Brainteaser)
[![arXiv](https://img.shields.io/badge/arXiv-brainteaser-firebrick)](https://arxiv.org/abs/2505.10844) 
[![PDF](https://img.shields.io/badge/PDF-silver)](https://arxiv.org/pdf/2505.10844)
```bibtex
@article{han2025creativity,
  title={Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models},
  author={Han, Simeng and Dai, Howard and Xia, Stephen and Zhang, Grant and Liu, Chen and Chen, Lichang and Nguyen, Hoang Huy and Mei, Hongyuan and Mao, Jiayuan and McCoy, R. Thomas},
  journal={Advances in neural information processing systems},
  year={2025}
}
```

</details>
