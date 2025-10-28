<p align="center">
  <img src="Logo.png" alt="Description" width="30%">
</p>

# Operational Planning of Hydrogen-Centric Companies

This repository accompanies the paper ‘**A Portfolio-Level Optimization Framework for Coordinated Market Participation and Operational Scheduling of Hydrogen-Centric Companies**’, presented at the 2025 IEEE International Conference on Energy Technologies for Future Grids. It has been developed as part of the **WinHy** project, funded by the Dutch Research Council (NWO) and Repsol S.A.

## 📝 Description
This repository provides the implementation of a portfolio-level optimization framework for hydrogen-centric companies that simultaneously operate across electricity, hydrogen, and green certificate markets. The model is designed to co-optimize operational scheduling and market participation for geographically distributed assets, including electrolyzers, renewable generation units, and energy storage systems. The framework is formulated as a Mixed-Integer Linear Programming (MILP) model and implemented in Python (3.12.5) using Pyomo.

---

## ✨ Key Features
- Multi-market integration: Co-optimizes participation in electricity, hydrogen (bundled and unbundled), and green certificate markets.  
- Portfolio-level coordination: Unlocks flexibility by centrally scheduling distributed assets across multiple sites, beyond individual asset operation.  
- Contractual heterogeneity: Supports both physical and virtual Power Purchase Agreements (PPAs) with take-as-produced structures.  
- Policy compliance: Incorporates company-level green hydrogen targets, certification rules, and clean energy temporal matching constraints.  
- Scalability: Applicable to hydrogen-centric companies of different sizes with multiple operational scenarios.  

---

## ⚙️ Model Highlights
- Implemented as a day-ahead operational planning model.  
- Objective function maximizes total company profit, considering hydrogen sales revenues, certificate transactions, electricity market exchanges, and PPA settlements.  
- Captures asset-level technical constraints (electrolyzers, energy storage, renewable generation).  
- Enables comparative analysis of different compliance strategies (per-site vs. portfolio-level enforcement).  

---

## 🧪 Case Study
The framework is demonstrated on a representative hydrogen-centric company (**H2FLEX**) operating five sites across Spain. Three operational setups are compared:  
- **Case 1**: Each electrolyzer operates independently with its own PPA and individual green hydrogen target constraints.  
- **Case 2**: PPAs are centrally dispatched among electrolyzers by the company operator, while green hydrogen target constraints are still enforced on each site individually.  
- **Case 3**: Both PPAs and green hydrogen targets are managed at the portfolio level by the company operator.     

---

## 📊 Key Results
- Centralized coordination enables up to a **2.42× increase in hydrogen production**.  
- Achieves a **9.4% reduction in daily operational costs**.  
- Portfolio-level enforcement improves flexibility, allowing **46.6% higher hydrogen production** while maintaining green hydrogen certification compliance.  

---

## 📂 Repo Structure

```
├─ H2FlexCo.ipynb                # Main Jupyter Notebook with the optimization model
├─ H2FlexCo.py                   # Python version of the Main Jupyter Notebook
├─ SimData.xlsx                  # Excel file containing the input simulation data
└─ requirements.txt              # List of required Python packages
├─ Cases/                    
│  ├─ Case_1.ipynb               # Decentralized site-level operation
│  ├─ Case_2.ipynb               # Centralized PPA dispatch
│  └─ Case_3.ipynb               # Full portfolio-level coordination with centralized policy enforcement
```

---

## 🚀 Requirements

Install the necessary Python libraries using:

```bash
pip install -r requirements.txt
```

---

## 📈 How to Run

1. Open `H2FlexCo.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure `SimData.xlsx` is in the same directory as the notebook.
3. Run all cells in the notebook to execute the model and generate results.

---

## 📦 Dependencies

The code uses the following libraries:
- `pyomo`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `openpyxl`

You may also need a solver like GLPK or IPOPT for Pyomo.

## 📚 Citations
If you use this repository in your work, please cite: 

*Mansouri, S. A., & Bruninx, K. (2025). A Portfolio-Level Optimization Framework for Coordinated Market Participation and Operational Scheduling of Hydrogen-Centric Companies. IEEE International Conference on Energy Technologies for Future Grids.*

---

## 📝 License

MIT License.
