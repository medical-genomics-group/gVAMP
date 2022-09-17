Currently supported input options to cpp version of the code:

| Input option | Description |
| --- | --- |
| `bed-file` | filepath to .bed file include the .bed extension |
| `phen-files` | path to file containing phenotype of interest (only 1 phenotype supportet at the moment) |
| `N` | number of individuals included in the inference process |
| `Mt` | total number of markers included in the infrence process |
| `out-dir` | output directory for the signal estimates |
| `out-name` | name of the output file |
| `iterations` | maximal number of iterations to be performed |
| `num-mix-comp` | number of gaussian mixture components used (including delta spike at zero) |
| `CG-max-iter` | maximal number of iteration used in conjugate gradient method for solving linear systems |
| `probs` | initial prior mixture coefficients (separated by comma, must sum up to 1) |
| `vars` | initial prior variances (separated by comma) |
| `rho` | initial value of damping factor |
| `EM-err-thr` | relative error threshold within expectation maximization |
| `EM-max-iter` | maximal number of iterations of expectation maximization procedure |
| `stop-criteria-thr` | relative error threshold within expectation maximization |
| `model` | regression model that describes a relationship between effect sizes and phenotypes ('linear' or 'bin_class') |
