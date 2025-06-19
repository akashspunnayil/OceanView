import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tempfile
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🌊 Ocean Data Viewer")

# ---- NetCDF Loader ----
@st.cache_data
def load_netcdf(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
        tmp_file.write(file_obj.read())
        tmp_path = tmp_file.name
    try:
        return xr.open_dataset(tmp_path, engine="netcdf4")
    except ValueError:
        st.warning("⚠️ Calendar decoding failed. Retrying with `decode_times=False`.")
        return xr.open_dataset(tmp_path, decode_times=False, engine="netcdf4")

# ---- Helper Functions ----
def find_coord_name(ds, keyword):
    for coord in ds.coords:
        if keyword.lower() in coord.lower():
            return coord
    return None

def try_decode_time(ds, time_var):
    time_vals = ds[time_var].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        return time_vals
    else:
        st.warning("⚠️ Time units not decoded. Approximating monthly time labels.")
        try:
            start = pd.Timestamp("2000-01-01")
            return pd.date_range(start, periods=len(time_vals), freq="MS")
        except Exception as e:
            st.error(f"❌ Time could not be approximated: {e}")
            return time_vals

# ---- File Upload ----
uploaded_file = st.file_uploader("📂 Upload a NetCDF file", type=["nc"])

if uploaded_file:
    ds = load_netcdf(uploaded_file)

    if ds is not None:
        st.success("✅ File loaded successfully.")
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

            # ---- Handle time properly ----
            time_sel = None
            if time_var:
                time_vals = try_decode_time(ds, time_var)
                time_sel = st.selectbox("🕒 Select Time", time_vals)

            # ---- Prepare subsetting arguments ----
            subset_kwargs = {
                lat_var: slice(*lat_range),
                lon_var: slice(*lon_range),
            }

            if time_sel is not None and time_var:
                time_dtype = ds[time_var].dtype
                if np.issubdtype(time_dtype, np.datetime64):
                    subset_kwargs[time_var] = np.datetime64(time_sel)
                elif np.issubdtype(time_dtype, np.number):
                    try:
                        time_index = list(time_vals).index(time_sel)
                        subset_kwargs[time_var] = time_index
                    except ValueError:
                        st.error("❌ Selected time not found in dataset index.")
                else:
                    subset_kwargs[time_var] = time_sel

            # ---- Subset and Plot ----
            try:
                # Step 1: select time separately if present
                if time_sel is not None and time_var:
                    ds_time_sel = ds[var].sel({time_var: time_sel}, method="nearest")
                else:
                    ds_time_sel = ds[var]
            
                # Step 2: slice lat/lon using slice objects
                data = ds_time_sel.sel({
                    lat_var: slice(*lat_range),
                    lon_var: slice(*lon_range)
                })
            
                # Plotting
                st.subheader("📍 Map View")
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
                data.squeeze().plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
                ax.coastlines()
                ax.set_title(f"{var} at {time_sel}" if time_sel is not None else var)
                st.pyplot(fig)
        
        except Exception as e:
            st.error(f"⚠️ Failed to subset and plot data: {e}")


                st.subheader("📍 Map View")
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
                data.squeeze().plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
                ax.coastlines()
                ax.set_title(f"{var} at {time_sel}" if time_sel is not None else var)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Failed to subset and plot data: {e}")
