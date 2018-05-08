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
* write access to the current directory (otherwise, output directories in download.py and *.sh
scripts must be changed)

## Reproducing Figures 1 and 5, and Table 2

Before running any of the analysis scripts, create the data/ directory and download votes files by running\
`$ python download.py`

To generate Figures 1 and 5 run\
`$ sh generate_figures.sh`\
The output is written to the figures/ directory.

For Table 2 run (may take several hours)\
`$ sh generate_table.sh`\
The output is written to the console.

For data-independent bounds (for comparison with Table 2), run\
`$ sh generate_table_data_independent.sh`\
The output is written to the console.

## Files in this directory

*   generate_figures.sh &mdash; Master script for generating Figures 1 and 5.

*   generate_table.sh &mdash; Master script for generating Table 2.

*   generate_table_data_independent.sh &mdash; Master script for computing data-independent
    bounds.

*   rdp_bucketized.py &mdash; Script for producing Figure 1 (right) and Figure 5 (right).

*   rdp_cumulative.py &mdash; Script for producing Figure 1 (middle) and Figure 5 (left).
   
*   smooth_sensitivity_table.py &mdash; Script for generating Table 2.

*   utility_queries_answered &mdash; Script for producing Figure 1 (left).

*   plot_partition.py &mdash; Script for producing partition.pdf, a detailed breakdown of privacy
costs for Confident-GNMax with smooth sensitivity analysis (takes ~50 hours).

*   plots_for_slides.py &mdash; Script for producing several plots for the slide deck. 

*   download.py &mdash; Utility script for populating the data/ directory.

*   plot_ls_q.py is not used.


All Python files take flags. Run script_name.py --help for help on flags.
