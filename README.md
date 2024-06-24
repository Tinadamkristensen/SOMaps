# Cognitive profiles across the psychosis continuum

This is a code base to reproduce our paper on _Cognitive profiles across the psychosis continuum_, available [ADD LINK]().

You can create a environment running the following command.
```bash
conda env create -f requirements.yml
```

(1) To run the PCA analysis:
```bash
# With all subjects
python PCA/pca_cogn.py --population ALL

# With Antipsychotic naive subjects only
python PCA/pca_cogn.py --population AP_naive
```

(2) To run the SOM analysis:
```bash
# With all subjects and standardization based on HC
python SOM/som_cogn.py --norm HC --population AP_naive

# With Antipsychotic naive subjects only and standardization based on HC
python SOM/som_cogn.py --norm HC --population ALL

# With all subjects and standardization based on entire opulation
python SOM/som_cogn.py --norm ALL --population ALL

# With Antipsychotic naive subjects only and standardization based on entire population
python SOM/som_cogn.py --norm ALL --population AP_naive

```
