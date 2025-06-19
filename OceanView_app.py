import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import io

st.title("🌊 Ocean Data Viewer")

import tempfile

@st.cache_data
def load_netcdf(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
        tmp_file.write(file_obj.read())
        tmp_path = tmp_file.name
    try:
        return xr.open_dataset(tmp_path, engine="netcdf4")
    except ValueError as e:
        st.warning("Fallback to decode_times=False due to calendar error.")
        return xr.open_dataset(tmp_path, decode_times=False, engine="netcdf4")


uploaded_file = st.file_uploader("Upload NetCDF file", type=["nc"])


if uploaded_file:
    ds = load_netcdf(uploaded_file)

# if uploaded_file:
#     file_bytes = io.BytesIO(uploaded_file.read())  # 🔁 This is the missing step
#     ds = load_netcdf(file_bytes)

    if ds is not None:
        st.success("✅ NetCDF file loaded successfully!")
        st.write("**Dimensions:**", ds.dims)
        st.write("**Variables:**", list(ds.data_vars))

    #ds = xr.open_dataset(uploaded_file)

    st.subheader("Dataset Dimensions and Variables")
    st.write("Dimensions:", ds.dims)
    st.write("Variables:", list(ds.data_vars.keys()))

    # Variable selection
    var = st.selectbox("Select a variable", list(ds.data_vars.keys()))

    # Latitude/Longitude selectors
    lat_range = st.slider("Latitude Range", float(ds.lat.min()), float(ds.lat.max()), 
                          (float(ds.lat.min()), float(ds.lat.max())))
    lon_range = st.slider("Longitude Range", float(ds.lon.min()), float(ds.lon.max()), 
                          (float(ds.lon.min()), float(ds.lon.max())))

    # Time selector (assumes time is available)
    if "time" in ds.coords:
        time_vals = ds.time.values
        time_sel = st.selectbox("Select Time", time_vals)

    # Subset data
    data = ds[var].sel(
        lat=slice(*lat_range),
        lon=slice(*lon_range),
        time=time_sel if "time" in ds.coords else None
    )

    # Plot
    st.subheader("📍 Map View")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    im = data.squeeze().plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
    ax.coastlines()
    st.pyplot(fig)

