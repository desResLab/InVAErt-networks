 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
# InVAErt networks: a data-driven framework for model synthesis and identifiability analysis

InVAErt networks are designed to perform emulation, inference, and identifiability analysis of physics-based parametric systems.

For additional information, please refer to the publication below:

[InVAErt networks: A data-driven framework for model synthesis and identifiability analysis](https://www-sciencedirect-com.proxy.library.nd.edu/science/article/pii/S0045782524001026), Guoxiang Grayson Tong, [Carlos A. Sing-Long Collao](https://www.ing.uc.cl/academicos-e-investigadores/carlos-alberto-sing-long-collao/), and [Daniele E. Schiavazzi](https://www3.nd.edu/~dschiava/).

### Description of the ```Tools``` folder
1. ```DNN_tools.py```: common functions for deep neural network modeling 
2. ```Data_generation.py```: functions for generating synthetic dataset of each numerical example
3. ```Model.py```: neural network modules 
4. ```NF_tools.py```: Specific functions used by the Real-NVP based normalizing flow model
5. ```Training_tools.py```: training and testing functions of the inVAErt networks
6. ```plotter.py```: common and specific plotter functions

### Current Jupyter notebooks:
1. ```Underdetermined_Linear_System.ipynb```: Section 4.1 of the paper. Study of an underdetermined linear system with non-trivial null space.
2. ```Single_sine_wave.ipynb```: Section 4.2 of the paper. Study of a simple nonlinear system: sine waves without periodicity.
3. ```Sine_Waves.ipynb```: Section 4.2 of the paper. Study of the former sine waves problem with periodicity.
4. ```RCR.ipynb```: Section 4.3 of the paper. Study of the non-identifiable three-element (R-C-R) Windkessel model.
5. ```Lotka-Volterra.ipynb```: Additional example, not in the paper. Study of the predator-prey model.

### Please stay tuned for more Jupyter Notebook tutorials!
Note: the Jupyter notebooks are created for illustration purposes thus the hyper-parameters are adjusted for swift and efficient execution. For more accurate results, we recommend running the code locally with fine-tuned hyper-parameters. Suggested hyper-parameters can be found in the appendix of the paper.

### Citation
Did you find this useful? Please cite us using:
```
@article{tong2024invaert,
  title={InVAErt networks: A data-driven framework for model synthesis and identifiability analysis},
  author={Tong, Guoxiang Grayson and Long, Carlos A Sing and Schiavazzi, Daniele E},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={423},
  pages={116846},
  year={2024},
  publisher={Elsevier}
}
```

### Recommended dependencies:
- ```Pytorch```: 2.4.1
- ```CUDA```: 11.8
- ```Python```: 3.10.12
- ```numpy```: 1.26.4
- ```scipy```: 1.12.0
- ```matplotlib```: 3.9.2
- ```mpi4py```: 4.0.0
