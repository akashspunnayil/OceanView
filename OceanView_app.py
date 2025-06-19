import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tempfile
import numpy as np
import pandas as pd
from xarray import decode_cf

st.set_page_config(layout="wide")
st.title("🌊 Ocean Data Viewer")

# ---- Safe NetCDF Loader ----
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

# ---- Helper Functions ----
def find_coord_name(ds, keyword):
    for coord in ds.coords:
        if keyword.lower() in coord.lower():
            return coord
    return None

def try_decode_time(ds, time_var):
    time_vals = ds[time_var].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        return time_vals, time_vals
    else:
        st.warning("⚠️ Time not decoded. Approximating time as monthly steps from 2000-01.")
        try:
            fake_time = pd.date_range("2000-01-01", periods=len(time_vals), freq="MS")
            return time_vals, fake_time
        except Exception as e:
            st.error(f"❌ Failed to create fake time labels: {e}")
            return time_vals, time_vals

# ---- File Upload ----
uploaded_file = st.file_uploader("📂 Upload a NetCDF file", type=["nc"])

if uploaded_file:
    ds = load_netcdf_safe(uploaded_file)

    if ds is not None:
        st.success("✅ File loaded successfully.")
        st.write("### 📦 Dataset Summary")
        st.code(ds.__repr__())
        st.write("**Dimensions:**", ds.dims)
        st.write("**Variables:**", list(ds.data_vars))
        
        var = st.selectbox("🔎 Select a variable to visualize", list(ds.data_vars.keys()))

        lat_var = find_coord_name(ds, "lat")
        lon_var = find_coord_name(ds, "lon")
        time_var = find_coord_name(ds, "time")

        if not lat_var or not lon_var:
            st.error("❌ Latitude or Longitude variable not found.")
        else:
            lat_vals = ds[lat_var].values
            lon_vals = ds[lon_var].values

            lat_range = st.slider("🌐 Latitude Range",
                                  float(lat_vals.min()), float(lat_vals.max()),
                                  (float(lat_vals.min()), float(lat_vals.max())))
            lon_range = st.slider("🗺️ Longitude Range",
                                  float(lon_vals.min()), float(lon_vals.max()),
                                  (float(lon_vals.min()), float(lon_vals.max())))

            time_sel = None
            if time_var:
                raw_time_vals, time_labels = try_decode_time(ds, time_var)
                time_sel = st.selectbox("🕒 Select Time", time_labels, key="select_time")

                # Map selection back to original raw time index (for datasets with decode_times=False)
                try:
                    time_index = list(time_labels).index(time_sel)
                    raw_time_value = raw_time_vals[time_index]
                except Exception:
                    raw_time_value = raw_time_vals[0]
            else:
                raw_time_value = None

            # ---- Subset the data ----
            try:
                # 1. Time selection (if applicable)
                if time_var and raw_time_value is not None:
                    ds_time_sel = ds[var].sel({time_var: raw_time_value}, method="nearest")
                else:
                    ds_time_sel = ds[var]

                # 2. Spatial slice
                data = ds_time_sel.sel({
                    lat_var: slice(*lat_range),
                    lon_var: slice(*lon_range)
                })

                # ---- Plotting ----
                st.subheader("📍 Map View")
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
                data.squeeze().plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
                ax.coastlines()
                ax.set_title(f"{var} at {time_sel}" if time_sel is not None else var)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Failed to subset or plot data: {e}")
