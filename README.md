# 🌊 OceanView: Interactive NetCDF Viewer App

**OceanView** is an advanced Streamlit-based web application for visualizing and exploring oceanographic datasets in NetCDF format. It is designed specifically for researchers working with 4D datasets (Time, Depth, Latitude, Longitude) and scalar ocean variables (e.g., temperature, salinity, oxygen, chlorophyll).

---

## 🎯 Key Objectives

- Provide an intuitive GUI for scientific exploration of NetCDF files.
- Enable flexible spatial, temporal, and depth-based slicing and plotting.
- Support vertical sections, Hovmöller diagrams, animations, and time series.
- Facilitate high-quality figure exports for publications.

---

## ✅ Key Features

### 🗂️ General
- Upload and preview **NetCDF** files.
- Auto-detect standard dimension names (`time`, `depth`, `lat`, `lon`).
- Select any scalar variable to visualize.
- Metadata display: variable units, dimensions, attributes.

---

## 🗺️ Spatial Plotting Modes

### 🔹 Mode 1: Single Time + Single Depth
- Plot a 2D horizontal map at a specific time and depth.
- Supports interactive zoom, panning, and map projections (Cartopy).

### 🔹 Mode 2: Time-Averaged + Single Depth
- Select a time range and plot the average at a single depth level.

### 🔹 Mode 3: Single Time + Depth-Averaged
- Select a depth range and plot spatial map at a single time after vertical averaging.

### 🔹 Mode 4: Time-Averaged + Depth-Averaged
- Average both time and depth ranges and plot the resulting spatial field.

### Plot Customization
- Select colormap, colorbar limits (manual/auto), font size.
- Enable land/sea masking.
- Adjust longitude/latitude ticks.
- Export plots as PNG.

---

## 🔍 Vertical Section Plotting

### Section Modes:
- **Z vs Longitude at fixed Latitude**
- **Z vs Latitude at fixed Longitude**
- **Z vs Longitude (averaged over Latitude band)**
- **Z vs Latitude (averaged over Longitude band)**

### Features:
- Select single time or average over time range.
- Customize depth limit, colormap, and labels.
- Export as static or interactive plot.

---

## 📈 Time Series Plotting Modes

### 1. Point-based Time Series
- Select a single Lat–Lon–Depth coordinate.
- Plot time evolution of the variable at that point.

### 2. Grid-Averaged Time Series
- Specify a lat-lon box and single depth.
- Plot time evolution of spatially averaged values.

### 3. Depth-Averaged Time Series
- Specify a depth range at a fixed lat-lon.
- Plot vertically averaged time series.

### 4. Full Spatiotemporal Average
- Specify both spatial and depth box.
- Plot completely averaged time series over the box.

---

## 🎞️ Time Animation (GIF)

- Select time range, depth, and variable.
- Generate animated GIF of temporal evolution.
- Control frame rate and animation speed.
- Option to export animation.

---

## 📊 Hovmöller Diagrams

### Modes:
- Longitude vs Time at fixed Lat–Depth
- Latitude vs Time at fixed Lon–Depth
- Depth vs Time at fixed Lat–Lon
- Depth vs Time (Grid-averaged)

### Features:
- Time range selection
- Customizable colormaps and labels
- Exportable plots

---

## 🧪 Interactive Features

- Toggle between static (Matplotlib) and interactive (Plotly) rendering.
- Optionally enable land/sea masking and tick styling.
- Use grid sliders or number inputs for precise control.

---

## 📦 Dependencies

- `streamlit`
- `xarray`, `netCDF4`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`, `plotly`
- `cartopy`
- `tempfile`, `PIL`

---

## ▶️ How to Launch

```bash
pip install -r requirements.txt
streamlit run OceanView_app.py

