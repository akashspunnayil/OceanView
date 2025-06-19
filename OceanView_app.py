import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

st.title("🌊 Ocean Data Viewer")

uploaded_file = st.file_uploader("Upload NetCDF file", type=["nc"])

if uploaded_file:
    ds = xr.open_dataset(uploaded_file)

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

