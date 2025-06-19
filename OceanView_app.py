import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tempfile
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🌊 Ocean Data Viewer")

# --- Safe NetCDF loader with fallback for time decoding errors ---
@st.cache_data
def load_netcdf_safe(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
        tmp_file.write(file_obj.read())
        tmp_path = tmp_file.name
    try:
        return xr.open_dataset(tmp_path, engine="netcdf4")
    except ValueError as e:
        if "unable to decode time units" in str(e) and "calendar 'NOLEAP'" in str(e):
            st.warning("⚠️ Time decoding failed. Retrying with decode_times=False...")
            return xr.open_dataset(tmp_path, decode_times=False, engine="netcdf4")
        else:
            raise

# --- Coordinate finder ---
def find_coord_name(ds, keyword):
    for coord in ds.coords:
        if keyword.lower() in coord.lower():
            return coord
    return None

# --- Time decoding fallback ---
def try_decode_time(ds, time_var):
    time_vals = ds[time_var].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        return time_vals, time_vals
    else:
        st.warning("⚠️ Time not decoded. Approximating monthly time from 2000-01.")
        try:
            fake_time = pd.date_range("2000-01-01", periods=len(time_vals), freq="MS")
            return time_vals, fake_time
        except Exception as e:
            st.error(f"❌ Failed to create fake time labels: {e}")
            return time_vals, time_vals

# --- File uploader ---
uploaded_file = st.file_uploader("📂 Upload a NetCDF file", type=["nc"])

if uploaded_file:
    ds = load_netcdf_safe(uploaded_file)

    if ds is not None:
        st.success("✅ File loaded successfully.")

        # # === Dataset structure printout (like Jupyter repr) ===
        # st.subheader("📄 Dataset Structure (like Jupyter)")
        # st.code(ds.__repr__(), language="python")

        #-------LEFT SIDE--------
        
        # === Layout split: controls on left, output on right ===
        left_col, right_col = st.columns([1, 2])  # or [1, 3] for wider output
        
        with left_col:
            var = st.selectbox("🔎 Variable", list(ds.data_vars.keys()))
        
            lat_var = find_coord_name(ds, "lat")
            lon_var = find_coord_name(ds, "lon")
            time_var = find_coord_name(ds, "time")
            depth_var = find_coord_name(ds, "depth")
            
            # 🔧 Manual fallback if any not found
            if not lat_var or not lon_var:
                st.warning("⚠️ Latitude or Longitude not automatically detected.")
                st.info("Refer to the dataset summary on the right to find actual coordinate names.")
                
            if not lat_var:
                lat_var = st.text_input("Enter Latitude Dimension Name", value="")
            if not lon_var:
                lon_var = st.text_input("Enter Longitude Dimension Name", value="")
            
            if not time_var:
                time_var = st.text_input("Enter Time Dimension Name (if available)", value="")
            
            if not depth_var:
                depth_var = st.text_input("Enter Depth Dimension Name (optional)", value="")

        
            if not lat_var or not lon_var:
                st.error("❌ Latitude or Longitude coordinate not found.")
            else:
                lat_vals = ds[lat_var].values
                lon_vals = ds[lon_var].values
        
                lat_range = st.slider("🌐 Latitude Range",
                                      float(lat_vals.min()), float(lat_vals.max()),
                                      (float(lat_vals.min()), float(lat_vals.max())))
                lon_range = st.slider("🗺️ Longitude Range",
                                      float(lon_vals.min()), float(lon_vals.max()),
                                      (float(lon_vals.min()), float(lon_vals.max())))
        
                if depth_var:
                    depth_vals = ds[depth_var].values
                    if len(depth_vals) == 1:
                        selected_depth = depth_vals[0]
                        st.info(f"🧭 Only one depth level: {selected_depth}")
                    else:
                        selected_depth = st.slider("🧭 Select Depth Level",
                                                   float(depth_vals.min()), float(depth_vals.max()),
                                                   float(depth_vals.min()))
                else:
                    selected_depth = None
        
                if time_var:
                    raw_time_vals, time_labels = try_decode_time(ds, time_var)
                    time_sel = st.selectbox("🕒 Select Time", time_labels, key="select_time")
                    try:
                        time_index = list(time_labels).index(time_sel)
                        raw_time_value = raw_time_vals[time_index]
                    except Exception:
                        raw_time_value = raw_time_vals[0]
                else:
                    raw_time_value = None

        
            with st.expander("🎨 Colorbar & Colormap Settings"):
                set_clim = st.checkbox("🔧 Manually set colorbar range")
            
                if set_clim:
                    vmin = st.number_input("Minimum value (vmin)", value=0.0)
                    vmax = st.number_input("Maximum value (vmax)", value=1.0)
                    step = st.number_input("Tick interval (optional)", value=0.1)
                else:
                    vmin, vmax, step = None, None, None
            
                cmap_choice = st.selectbox(
                    "🎨 Choose a colormap",
                    options=sorted([
                        "viridis", "plasma", "inferno", "magma", "cividis",
                        "jet", "turbo", "coolwarm", "RdBu_r", "YlGnBu", "BrBG", "bwr"
                    ]),
                    index=0
                )

        #-------RIGHT SIDE--------

        with right_col:
            st.subheader("📄 Dataset Structure")
            st.code(ds.__repr__(), language="python")
        
            # === Subsetting ===
            try:
                if time_var and raw_time_value is not None:
                    ds_sel = ds[var].sel({time_var: raw_time_value}, method="nearest")
                else:
                    ds_sel = ds[var]
        
                if depth_var and selected_depth is not None:
                    ds_sel = ds_sel.sel({depth_var: selected_depth}, method="nearest")
        
                subset_kwargs = {
                    lat_var: slice(*lat_range),
                    lon_var: slice(*lon_range)
                }
                data = ds_sel.sel(subset_kwargs)
        
                # === Plotting ===
                st.subheader("🗺️ Map View")
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
                # data.squeeze().plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
                plot_kwargs = {
                    "ax": ax,
                    "transform": ccrs.PlateCarree(),
                    "cmap": cmap_choice,
                    "add_colorbar": True
                }
                
                if set_clim:
                    plot_kwargs["vmin"] = vmin
                    plot_kwargs["vmax"] = vmax
                
                data.squeeze().plot.pcolormesh(**plot_kwargs)

                ax.coastlines()
                title = f"{var}"
                if time_var: title += f" | Time: {time_sel}"
                if depth_var: title += f" | Depth: {selected_depth} m"
                ax.set_title(title)
                st.pyplot(fig)
        
            except Exception as e:
                st.error(f"⚠️ Failed to subset or plot data: {e}")

