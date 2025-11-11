import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
import os

# --- ASK USER FOR LOCAL FILE PATH ---
file_path = input("Enter the local path of your NMR CSV file: ").strip()

# --- LOAD DATA ---
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Cannot find file: {file_path}")

data = pd.read_csv(file_path)

# --- USE FIRST TWO COLUMNS AS X AND Y ---
x = data.iloc[:, 0].to_numpy()
y = data.iloc[:, 1].to_numpy()
print(f"Using columns: X = {data.columns[0]}, Y = {data.columns[1]}")

# --- SORT DATA (important if ppm decreases) ---
idx = np.argsort(x)
x, y = x[idx], y[idx]

# --- INTEGRATE OVER FULL RANGE ---
area_y  = simps(y, x)          # ∫ y dx
area_xy = simps(x * y, x)      # ∫ x*y dx
centroid = area_xy / area_y if area_y != 0 else np.nan

# --- PLOT RESULTS ---
plt.figure(figsize=(8,5))
plt.plot(x, y, color='black', lw=1.2, label="NMR Spectrum")
plt.fill_between(x, y, color='orange', alpha=0.3, label="Integrated region")

plt.axvline(centroid, color='red', linestyle='--', lw=1.2,
            label=f"Centroid = {centroid:.3f}")
plt.text(centroid, max(y)*0.8, f"⟨x⟩ = {centroid:.3f}",
         color='red', ha='left', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel("Chemical Shift (ppm)")
plt.ylabel("Intensity (a.u.)")
plt.title("Full NMR Spectrum Integration and Centroid")
plt.legend()
plt.tight_layout()
plt.show()

# --- PRINT RESULTS ---
print("\nFull-Range Integration Results:")
print(f"∫y dx     = {area_y:.6f}")
print(f"∫x·y dx   = {area_xy:.6f}")
print(f"⟨x⟩ (ppm) = {centroid:.6f}")
