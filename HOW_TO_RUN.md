# How to Run — Astronaut Health Monitoring System
**NASA Space Apps Challenge 2026 (Test Prototype)**

This guide provides step-by-step instructions to set up the project from scratch ("Phase Zero") inside a clean Python virtual environment (`venv`) on both **Windows** and **macOS / Linux**.

---

## Prerequisites (Phase Zero)

Before doing anything, ensure you have **Python 3.10+** (Python 3.10, 3.11, or 3.12 recommended) installed on your system.

### Check Python Version
Open your terminal (PowerShell / Command Prompt on Windows, Terminal on macOS) and run:

- **Windows:**
  ```powershell
  python --version
  ```
- **macOS / Linux:**
  ```bash
  python3 --version
  ```
If not installed:
- **Windows:** Download and install from [python.org](https://www.python.org/downloads/) (make sure to check *"Add python.exe to PATH"* during installation).
- **macOS:** Install via Homebrew: `brew install python` or download the installer from [python.org](https://www.python.org/downloads/).

---

## 1. Setup on Windows (PowerShell or Command Prompt)

### Step 1: Open Terminal & Navigate to Project Directory
```powershell
cd "D:\Actual Project Files\Training\Python\Machine Learning\Anomaly Detection"
```
*(Replace the path with your local folder path if cloning or running elsewhere).*

### Step 2: Create a Virtual Environment (`venv`)
Create a dedicated virtual environment named `.venv`:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
- In **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If you encounter an execution policy error in PowerShell, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then rerun the activate command).*
- In **Command Prompt (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

Once activated, your terminal prompt will show `(.venv)`.

### Step 4: Upgrade `pip` & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Setup on macOS / Linux (Terminal)

### Step 1: Open Terminal & Navigate to Project Directory
```bash
cd "/path/to/Anomaly Detection"
```

### Step 2: Create a Virtual Environment (`venv`)
```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```bash
source .venv/bin/activate
```
Once activated, your terminal prompt will show `(.venv)`.

### Step 4: Upgrade `pip` & Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. How to Run the System

Once your virtual environment is active and dependencies are installed, you have three execution workflows available:

### Option A: Run the Primary Interactive Notebook (Recommended)

The notebook `notebooks/astronaut_health_monitor.ipynb` is the primary interactive demonstration surface.

1. **Register the kernel and launch Jupyter:**
   ```bash
   # Both Windows and Mac (inside active .venv)
   python -m ipykernel install --user --name=astro-health --display-name="Python (.venv)"
   jupyter notebook notebooks/astronaut_health_monitor.ipynb
   ```
   Or launch Jupyter Lab:
   ```bash
   jupyter lab
   ```

2. **Run in VS Code / Cursor / PyCharm:**
   - Open the workspace folder in your editor.
   - Open `notebooks/astronaut_health_monitor.ipynb`.
   - Click the Kernel selector in the top-right corner.
   - Select your `.venv` Python interpreter.
   - Click **Run All** to execute all 22 sections from top to bottom.

---

### Option B: Run the Automated Test Suite

To verify that all modules, algorithms, and edge cases function correctly:

```bash
# Both Windows and Mac (inside active .venv)
python -m pytest tests/test_pipeline.py -v
```

Expected output:
```text
============================= 42 passed in 2.15s ==============================
```

---

### Option C: Run Headless Execution / Regeneration

If you want to re-execute or regenerate the complete pipeline from the command line without opening a browser or UI:

1. **Regenerate the notebook structure:**
   ```bash
   python build_notebook.py
   ```

2. **Execute all code cells headlessly and re-save model artifacts:**
   ```bash
   python -c "
   import json, os
   from pathlib import Path

   nb_path = Path('notebooks/astronaut_health_monitor.ipynb')
   with open(nb_path, 'r', encoding='utf-8') as f:
       nb = json.load(f)

   os.chdir('notebooks')
   import matplotlib
   matplotlib.use('Agg')

   gl = {'__name__': '__main__'}
   for i, cell in enumerate(c for c in nb['cells'] if c['cell_type'] == 'code'):
       exec(''.join(cell['source']), gl)
   print('Complete pipeline executed headlessly! Artifacts updated in artifacts/')
   "
   ```

---

## 4. Verifying Outputs & Artifacts

After execution, all trained models, calibration metrics, evaluation CSVs, and plots are saved in the `artifacts/` folder:

- **Serialized Models (`.joblib`):**
  - `artifacts/preprocessor.joblib` — Fitted RobustScaler & feature metadata.
  - `artifacts/robust_baseline.joblib` — Baseline robust MAD medians.
  - `artifacts/isolation_forest.joblib` — Fitted Isolation Forest model.
  - `artifacts/autoencoder.joblib` — Fitted reconstruction model.
- **Evaluation Metrics:**
  - `artifacts/evaluation_results.csv` — Synthetic anomaly detection scores.
- **Visualization Figures (`.png`):**
  - `artifacts/dashboard.png` — Multi-panel executive summary dashboard.
  - `artifacts/eda_features_over_time.png` — HRV signals across experimental conditions.
  - `artifacts/feature_correlation.png` — Feature correlation matrix heatmap.
  - `artifacts/baseline_scores.png` — MAD robust Z-score timeline.
  - `artifacts/composite_scores.png` — Combined ensemble score timeline.
  - `artifacts/model_comparison.png` — Comparative detection metrics bar chart.
  - `artifacts/scores_by_condition.png` — Anomaly score distributions by condition.

---

## 5. Deactivating the Virtual Environment

When you are done working:
```bash
deactivate
```
