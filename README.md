This package loads CPT data, filters out unusable data,
and calculates Vs30 values for each CPT using several 
correlations. 

The calculations and CPT class were implemented by Joel Ridden
in the `vs_calc` package. 

The loading and filtering of input data was adapted from 
earlier work by Sung Bae in the `cpt2vs30` package.

To run this package, first configure the input parameters 
by editing the `config.yaml` file. Then run the `main.py` script as 

```python main.py```