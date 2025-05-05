<p align="center">
  <img src="Logo.png" alt="Description" width="30%">
</p>

# H2FlexCo: Hydrogen Flexibility Coordination Model


This repository contains a Pyomo-based energy system optimization model that utilizes flexibility resources to coordinate hydrogen production and consumption. The model uses input data from an Excel file and visualizes results with plots.

## 📂 Files

- `H2FlexCo.ipynb`: Main Jupyter Notebook with the optimization model.
- `SimData.xlsx`: Excel file containing the input simulation data.
- `requirements.txt`: List of required Python packages.

## 🚀 Requirements

Install the necessary Python libraries using:

```bash
pip install -r requirements.txt
```

## 📈 How to Run

1. Open `H2FlexCo.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure `SimData.xlsx` is in the same directory as the notebook.
3. Run all cells in the notebook to execute the model and generate results.

## 📦 Dependencies

The code uses the following libraries:
- `pyomo`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `pickle`

You may also need a solver like GLPK or IPOPT for Pyomo.

## 📝 License

MIT License.
