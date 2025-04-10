# 🏠 DC Affordable Housing Data Analysis

This project contains a data analysis notebook focused on affordable housing in Washington, D.C. It explores trends, affordability, and availability across wards and neighborhoods using open housing datasets.

---

## 📊 Overview

The primary objectives of this analysis are to:

- Understand the distribution of affordable housing in DC
- Analyze pricing, unit availability, and AMI (Area Median Income) coverage
- Visualize disparities across wards and neighborhoods
- Provide insights to assist with urban planning and housing policy

---

## 📁 Project Structure


---

## 🧾 Dataset

The dataset typically includes the following fields:

- `Project_Name`
- `Address`
- `Ward`
- `Units_Affordable`
- `Total_Units`
- `AMIs_Served`
- `Developer`
- `Completion_Year`
- `Latitude`, `Longitude`

**Source:** [Open Data DC](https://opendata.dc.gov/datasets/DCGIS::affordable-housing/about)
---

## ⚙️ Tech Stack

- Python 3.9+
- **Pandas**, **NumPy** – data wrangling
- **Matplotlib**, **Seaborn** – data visualization

---

## ▶️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/WiseTofu/TSADataScienceVA21173-1.git
   cd TSADataScienceVA21173-1
2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

4. **Run the analysis**
   ```bash
   python affhous.py
   ```

## 📊 Output

The script will:
- Display AMI distribution percentages by ward
- Show total affordable units per ward
- Generate visualizations:
  - Stacked bar chart of AMI levels by ward
  - Bar chart of total affordable units by ward

## 📝 Notes

- Requires Python 3.9 or higher
- Make sure `Affordable_Housing.csv` is in the project root directory
- Generated plots will appear in a new window
