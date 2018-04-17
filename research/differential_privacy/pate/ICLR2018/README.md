Scripts in support of the paper "Scalable Private Learning with PATE" by Nicolas
Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, Ulfar
Erlingsson (ICLR 2018, https://arxiv.org/abs/1802.08908).


### Requirements

* Python, version &ge; 2.7
* absl (see [here](https://github.com/abseil/abseil-py), or just type `pip install absl-py`)
* matplotlib
* numpy
* scipy
* sympy (for smooth sensitivity analysis)  
* write access to current directory (otherwise, output directories in download.py and *.sh scripts
must be changed)

## Reproducing Figures 1 and 5, and Table 2

Before running any of the analysis scripts, create the data/ directory and download votes files by running\
`$ python download.py`

To generate Figures 1 and 5 run\
`$ sh generate_figures.sh`\
The output is written to the figures/ directory.

For Table 2 run (may take several hours)\
`$ sh generate_table.sh`\
The output is written to the console.

For data-independent bounds (for comparing with Table 2), run\
`$ sh generate_table_data_independent.sh`\
The output is written to the console.

## Files in this directory

*   generate_figures.sh --- Master script for generating Figures 1 and 5.

*   generate_table.sh --- Master script for generating Table 2.

*   generate_table_data_independent.sh --- Master script for computing data-independent
    bounds.

*   rdp_bucketized.py --- Script for producing Figures 1 (right) and 5 (right).

*   rdp_cumulative.py --- Script for producing Figure 1 (left, middle), Figure 5
    (left), and partition.pdf (a detailed breakdown of privacy costs per
    source).
   
*   smooth_sensitivity_table.py --- Script for generating Table 2.

*   rdp_flow.py and plot_ls_q.py are currently not used.

*   download.py --- Utility script for populating the data/ directory.


All Python files take flags. Run script_name.py --help for help on flags.
