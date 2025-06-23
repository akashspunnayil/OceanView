# Streamlit OceanView - Data Viewer & Plotter

import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import tempfile
import io
import os
import plotly.graph_objects as go
    
st.set_page_config(layout="wide")
st.title("🌊 Ocean Viewer (NetCDF)")


# --- Safe NetCDF loader with fallback for time decoding errors ---
# @st.cache_data
# def load_netcdf_safe(file_obj):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
#         tmp_file.write(file_obj.read())
#         tmp_path = tmp_file.name
#     try:
#         return xr.open_dataset(tmp_path, engine="netcdf4")
#     except ValueError as e:
#         if "unable to decode time units" in str(e) and "calendar 'NOLEAP'" in str(e):
#             st.warning("⚠️ Time decoding failed. Retrying with decode_times=False...")
#             return xr.open_dataset(tmp_path, decode_times=False, engine="netcdf4")
#         else:
#             raise

@st.cache_data
def load_netcdf_safe(file_obj):
    import tempfile
    import xarray as xr

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    try:
        return xr.open_dataset(tmp_path)
    except ValueError as e:
        if "unable to decode time units" in str(e) and "calendar 'NOLEAP'" in str(e):
            st.warning("⚠️ Time decoding failed. Retrying with decode_times=False...")
            return xr.open_dataset(tmp_path, decode_times=False)
        else:
            raise

def load_netcdf_safe_from_path(path):
    try:
        return xr.open_dataset(path, engine="scipy") #netcdf4
    except ValueError as e:
        if "unable to decode time units" in str(e) and "calendar 'NOLEAP'" in str(e):
            st.warning("⚠️ Time decoding failed. Retrying with decode_times=False...")
            return xr.open_dataset(path, decode_times=False, engine="scipy") #netcdf4
        else:
            raise

def find_coord_from_dims(da, keyword):
    for dim in da.dims:
        if keyword.lower() in dim.lower():
            return dim
    return None

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

#-------------------------------------------------------------------------------------------------------------------#
def detect_coord_names(dataarray):
    # Known patterns (case-insensitive)
    candidates = {
        "latitude": ["lat", "latitude"],
        "longitude": ["lon", "longitude"],
        "depth": ["depth", "depth1_1", "DEPTH", "z"],
        "time": ["time", "TIME"]
    }

    # Match from dataarray coords
    found = {}
    coords_lower = {k.lower(): k for k in dataarray.coords}

    for standard_name, options in candidates.items():
        for name in options:
            if name.lower() in coords_lower:
                found[standard_name] = coords_lower[name.lower()]
                break
        else:
            found[standard_name] = None  # Not found

    return found

def scale_dataarray(dataarray, op, val):
    if op == "*":
        return dataarray * val
    elif op == "/":
        return dataarray / val
    elif op == "+":
        return dataarray + val
    elif op == "-":
        return dataarray - val
    return dataarray




# ------------------------------------------------ File uploader ---------------------------------------------- #
# uploaded_file = st.file_uploader("📂 Upload a NetCDF file", type=["nc"])

# Mode selection
mode = st.radio("Select mode", ["Upload file (Web App)", "Use local file (Download Desktop App)"])

if mode == "Use local file (desktop only)":
    file_path = st.text_input("Enter full path to NetCDF file")
    if file_path:
        if os.path.exists(file_path):
            try:
                ds = xr.open_dataset(file_path)
                st.success("✅ File loaded from local path.")
                # st.write(ds)
            except Exception as e:
                st.error(f"❌ Failed to open NetCDF: {e}")
        else:
            st.error("❌ File does not exist.")
else:
    # uploaded_file = st.file_uploader("### 📂 Upload a NetCDF file", type=["nc"])
    st.markdown("#### 📂 Upload a NetCDF file")
    uploaded_file = st.file_uploader("Upload file", type=["nc"], label_visibility="collapsed")

    if uploaded_file:
        try:
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            #     tmp.write(uploaded_file.read())
            #     ds = xr.open_dataset(tmp.name)
            ds = load_netcdf_safe(uploaded_file)

            # st.success("✅ File loaded from uploaded file.")
            # st.write(ds)

            if ds is not None:
                st.success("✅ File loaded successfully.")

                # Detect plot-compatible variables
                def is_plot_compatible(da):
                    dims = set(da.dims)
                    return any("lat" in d.lower() for d in dims) and any("lon" in d.lower() for d in dims)

                plot_vars = {v: ds[v] for v in ds.data_vars if is_plot_compatible(ds[v])}
                if not plot_vars:
                    st.error("❌ No valid spatial variables (lat/lon) found.")
                    st.stop()

                        
                #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                #--------------------------LEFT SIDE - INPUTS-------------------------------------#
                #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                left_col, right_col = st.columns([1, 2])

                with left_col:

                    st.markdown("#### 🔎 Variable")
                    var = st.selectbox("Variable", list(plot_vars.keys()), label_visibility="collapsed")

                    with st.expander("🧮 Apply Scaling to Variable Values (Optional)"):
                        apply_scaling = st.checkbox("Apply arithmetic scaling?")
                        if apply_scaling:
                            scale_op = st.selectbox("Operation", ["*", "/", "+", "-"], index=0)
                            scale_val = st.number_input("Scale Value", value=1.0, step=0.1)

                    # ds_sel = ds[var]
    
                    ds_sel = ds[var]
                    if apply_scaling:
                        ds_sel = scale_dataarray(ds_sel, scale_op, scale_val)
    
    
                    # Ensure dims are coords
                    for d in ds_sel.dims:
                        if d not in ds_sel.coords and d in ds.coords:
                            ds_sel = ds_sel.assign_coords({d: ds[d]})
                            
                    time_var = find_coord_from_dims(ds_sel, "time")
                    depth_var = find_coord_from_dims(ds_sel, "depth")
                    lat_var = find_coord_from_dims(ds_sel, "lat")
                    lon_var = find_coord_from_dims(ds_sel, "lon")

                    if lat_var and ds[lat_var][0] > ds[lat_var][-1]:
                        ds = ds.sortby(lat_var)
    
                    if "missing_value" in ds[var].attrs:
                        mv = ds[var].attrs["missing_value"]
                        if abs(mv) > 1e10:
                            ds[var] = ds[var].where(ds[var] != mv)
        
                    if time_var and not np.issubdtype(ds[time_var].dtype, np.datetime64):
                        try:
                            ds = ds.assign_coords({time_var: pd.date_range("2000-01-01", periods=ds.dims[time_var], freq="MS")})
                        except Exception as e:
                            st.error(f"❌ Time decoding failed: {e}")
        
                    if not lat_var:
                        lat_var = st.text_input("Enter Latitude Dimension Name", value="")
                    if not lon_var:
                        lon_var = st.text_input("Enter Longitude Dimension Name", value="")
                    if not time_var:
                        time_var = st.text_input("Enter Time Dimension Name (if available)", value="")
                    if not depth_var:
                        depth_var = st.text_input("Enter Depth Dimension Name (optional)", value="")
        
                    if not lat_var.strip() or not lon_var.strip():
                        st.error("❌ Latitude or Longitude coordinate not found.")
                        st.stop()
        
                    lat_vals = ds[lat_var].values
                    lon_vals = ds[lon_var].values


                    # --- Define bounds
                    lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
                    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
                    
                    # --- Initialize default values in session_state
                    default_coords = {
                        "north_lat": lat_max,
                        "south_lat": lat_min,
                        "west_lon": lon_min,
                        "east_lon": lon_max,
                    }
                    for key, val in default_coords.items():
                        if key not in st.session_state:
                            st.session_state[key] = val
                    
                    # --- Initialize reset flag
                    if "reset_coords" not in st.session_state:
                        st.session_state.reset_coords = False
                    
                    # --- Reset button
                    if st.button("🔄 Reset to Full Extent"):
                        st.session_state.reset_coords = True
                        st.rerun()
                    
                    # --- Handle reset before rendering widgets
                    if st.session_state.reset_coords:
                        for key, val in default_coords.items():
                            st.session_state[key] = val
                        st.session_state.reset_coords = False
                        st.rerun()
                    
                    # --- Region & Depth Selection UI
                    st.markdown("#### 🌐 Region & Depth Selection")
                    
                    # -- Top Row: North Latitude (centered)
                    cols_north = st.columns([1, 1, 1])
                    with cols_north[1]:
                        north_lat = st.number_input(
                            "⬆️ Lat Max",
                            min_value=lat_min,
                            max_value=lat_max,
                            value=st.session_state["north_lat"],
                            key="north_lat"
                        )
                    
                    # -- Middle Row: West and East Longitude
                    cols_mid = st.columns([1, 1])
                    with cols_mid[0]:
                        west_lon = st.number_input(
                            "⬅️ Lon Min",
                            min_value=lon_min,
                            max_value=lon_max,
                            value=st.session_state["west_lon"],
                            key="west_lon"
                        )
                    with cols_mid[1]:
                        east_lon = st.number_input(
                            "➡️ Lon Max",
                            min_value=lon_min,
                            max_value=lon_max,
                            value=st.session_state["east_lon"],
                            key="east_lon"
                        )
                    
                    # -- Bottom Row: South Latitude (centered)
                    cols_south = st.columns([1, 1, 1])
                    with cols_south[1]:
                        south_lat = st.number_input(
                            "⬇️ Lat Min",
                            min_value=lat_min,
                            max_value=lat_max,
                            value=st.session_state["south_lat"],
                            key="south_lat"
                        )
                    
                    # -- Final lat/lon range
                    lat_range = (south_lat, north_lat)
                    lon_range = (west_lon, east_lon)

                    
                    
                    
                        
                    # Use left_col for plot selection checkboxes
                    with st.expander("🗺️ Select Plot Options", expanded=True):
                        show_spatial_map = st.checkbox("Spatial Map")
                        show_interactive_spatial_map = st.checkbox("Spatial Interactive Map")
                        show_time_animation = st.checkbox("Spatial Map - Time Animation")
                        show_vertical_section = st.checkbox("Vertical Section")
                        show_interactive_vertical_section = st.checkbox("Interactive Vertical Section")
                        show_timeseries_plot = st.checkbox("Time Series Plot")
                        show_vertical_profile = st.checkbox("Vertical Profile")
                        show_interactive_vertical_profile =  st.checkbox("Vertical Interactive Profile ")
                        show_hovmoller = st.checkbox("Hovmöller Diagram")
        
                            
                    if show_spatial_map or show_vertical_section or show_time_animation or show_interactive_spatial_map:
                        with st.expander("🌍 Land/Sea Masking"):
                            mask_land = st.checkbox("Mask Land", value=False)
                            mask_sea = st.checkbox("Mask Ocean", value=False)
                            mask_color = st.selectbox("Mask Color", ["lightgray", "gray", "black", "white", "skyblue", "khaki", "coral", "forestgreen"])
                    
                    def reset_colorbar_settings():
                        st.session_state["set_clim"] = False
                        for key in ["vmin", "vmax", "step", "cmap_choice"]:
                            st.session_state.pop(key, None)
                        st.session_state["cmap_choice"] = "viridis"
        
                    if show_spatial_map or show_vertical_section or show_interactive_vertical_section or show_time_animation or show_interactive_spatial_map or show_hovmoller:
                        with st.expander("🎨 Colorbar & Colormap Settings"):
                            cols_colorbar = st.columns([2, 1])
                            with cols_colorbar[0]:
                                set_clim = st.checkbox("🔧 Manually set colorbar range", key="set_clim")
                                vmin = st.number_input("Minimum value (vmin)", value=0.0) if set_clim else None
                                vmax = st.number_input("Maximum value (vmax)", value=1.0) if set_clim else None
                                step = st.number_input("Tick interval (optional)", value=0.1) if set_clim else None
                                cmap_choice = st.selectbox("🎨 Choose a colormap", sorted(["viridis", "plasma", "inferno", "magma", "cividis", "jet", "turbo", "coolwarm", "RdBu_r", "YlGnBu", "BrBG", "bwr"]))
                            with cols_colorbar[1]:
                                # st.button("🔄 Reset", on_click=reset_colorbar_settings)
                                st.button("🔄 Reset", on_click=reset_colorbar_settings, key="reset_colorbar_btn")
        
                    # def reset_plot_labels(time_sel_value=None, depth_value=None):
                    #     # title = var
                    #     # if time_sel_value:
                    #     #     title += f" | {time_sel_value}"
                    #     # if depth_value is not None:
                    #     #     title += f" | {depth_value} m"
                    #     # st.session_state["plot_title"] = title
                    #     st.session_state["xlabel"] = "Longitude"
                    #     st.session_state["ylabel"] = "Latitude"
                    #     st.session_state["cbar_label"] = var
        
                    with st.expander("🖊️ Plot Custom Labels"):
                        label_cols = st.columns([2, 1])
                        with label_cols[0]:
                            plot_title = st.text_input("📌 Plot Title", value=var, key="plot_title")
                            xlabel = st.text_input("🧭 X-axis Label", value="Longitude", key="xlabel")
                            ylabel = st.text_input("🧭 Y-axis Label", value="Latitude", key="ylabel")
                            cbar_label = st.text_input("🎨 Colorbar Label", value=var, key="cbar_label")
                        # with label_cols[1]:
                            # st.button("🔄 Reset", on_click=reset_plot_labels)
                            # st.button("🔄 Reset", on_click=reset_plot_labels, key="reset_plot_labels_btn")
                            # st.button("🔄 Reset", on_click=lambda: reset_plot_labels(time_sel, selected_depth), key="reset_plot_labels_btn")
        
        
                    def reset_tick_settings():
                        st.session_state.pop("manual_ticks", None)
                        for key in ["xtick_step", "ytick_step"]:
                            st.session_state.pop(key, None)
                            
                    if show_spatial_map or show_time_animation or show_interactive_spatial_map:
                        with st.expander("📏 Axis Tick Settings"):
                            tick_cols = st.columns([2, 1])
                            with tick_cols[0]:
                                manual_ticks = st.checkbox("🔧 Manually set tick intervals", key="manual_ticks")
                                # xtick_step = st.number_input("Longitude Tick Interval (°)", min_value=0.1, max_value=60.0, value=10.0, step=1.0) if manual_ticks else None
                                # ytick_step = st.number_input("Latitude Tick Interval (°)", min_value=0.1, max_value=60.0, value=5.0, step=1.0) if manual_ticks else None
                                xtick_step = st.number_input(
                                    "Longitude Tick Interval (°)", 
                                    min_value=0.1, max_value=60.0, value=10.0, step=1.0, 
                                    key="xtick_step"
                                ) if manual_ticks else None
                                
                                ytick_step = st.number_input(
                                    "Latitude Tick Interval (°)", 
                                    min_value=0.1, max_value=60.0, value=5.0, step=1.0, 
                                    key="ytick_step"
                                ) if manual_ticks else None
            
                            with tick_cols[1]:
                                # st.button("🔄 Reset", on_click=reset_tick_settings)
                                st.button("🔄 Reset", on_click=reset_tick_settings, key="reset_tick_settings_btn")
                    
                    # Define the callback function
                    def reset_font():
                        st.session_state["font_family"] = "DejaVu Sans"
                    
                    # Font settings block
                    import os
                    import matplotlib.font_manager as fm
                    
                    # Font settings block
                    with st.expander("🖋️ Plot Font Settings"):
                        font_family = st.selectbox(
                            "Font Family",
                            options=[
                                "DejaVu Sans",
                                "Arial",
                                "Times New Roman",
                                "Courier New",
                                "Calibri Light",
                            ],
                            index=0,
                            key="font_family"
                        )
                    
                        # Try to load custom font from fonts/ directory
                        font_path = f"fonts/{font_family}.ttf"
                        if os.path.exists(font_path):
                            fm.fontManager.addfont(font_path)
                            resolved_font = fm.FontProperties(fname=font_path).get_name()
                            plt.rcParams["font.family"] = resolved_font
                            st.success(f"✅ Custom font applied: {resolved_font}")
                        else:
                            plt.rcParams["font.family"] = font_family
                            st.warning(f"⚠️ '{font_family}' .ttf not found in /fonts. Using system font fallback.")
                    
                        # Reset button
                        def reset_font():
                            st.session_state["font_family"] = "DejaVu Sans"
                    
                        st.button("Reset Font to Default", on_click=reset_font, key="reset_font_btn")
        
        
                    with st.expander("💾 Save Plot Options"):
                        save_format = st.selectbox("Select file format", ["png", "jpg", "pdf", "svg", "tiff"], index=0)
                        dpi_value = st.number_input("DPI (dots per inch)", min_value=50, max_value=600, value=150, step=10)
                        save_btn = st.button("💾 Save & Download Plot")

                #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                #------------------------RIGHT SIDE - OUTPUTS-------------------------------------#
                #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                with right_col:
                    # st.subheader("📄 Dataset Structure")
                    # st.code(ds.__repr__(), language="python")
                    with st.expander("📄 Dataset Structure"):
                        st.code(ds.__repr__(), language="python")

        
                    # # --- Subset and Plot ---
                    # subset_kwargs = {}
                    # if time_var and time_var in ds_sel.dims and raw_time_value is not None:
                    #     ds_sel = ds_sel.sel({time_var: raw_time_value}, method="nearest")
                    # if depth_var and depth_var in ds_sel.dims and selected_depth is not None:
                    #     ds_sel = ds_sel.sel({depth_var: selected_depth}, method="nearest")
                    # if lat_var in ds_sel.dims:
                    #     subset_kwargs[lat_var] = slice(*lat_range)
                    # if lon_var in ds_sel.dims:
                    #     subset_kwargs[lon_var] = slice(*lon_range)
                    
                    # data = ds_sel.sel(subset_kwargs)

                    
                    # =============================
                    # --- Advanced Slicing for Plot Mode ---
                    # =============================
                    data = ds[var]
                    data = data.sel({lat_var: slice(*lat_range), lon_var: slice(*lon_range)})
                
                    
                #++++++++++++++++++++++++++++                    
                        
                    #---------------------------------Normal Spatial Map View----------------------------------------------------------#
            
                    if show_spatial_map:
                        # -- Plot Mode Selection
                        plot_mode = st.radio("🧭 Select Plot Mode", [
                            "Single Time + Single Depth",
                            "Time Range Avg + Single Depth",
                            "Single Time + Depth Range Avg",
                            "Time Range Avg + Depth Range Avg"
                        ])

                        
                        # -- Depth Input
                        depth_vals = ds[depth_var].values if depth_var else None
                        if depth_var:
                            if "Depth Range Avg" in plot_mode:
                                col1, col2 = st.columns(2)
                                with col1:
                                    dmin = st.number_input("Min Depth", float(depth_vals.min()), float(depth_vals.max()), value=0.0, key="depth_min")
                                with col2:
                                    dmax = st.number_input("Max Depth", float(depth_vals.min()), float(depth_vals.max()), value=200.0, key="depth_max")
                            else:
                                selected_depth = st.number_input(
                                    "Depth (m)", float(depth_vals.min()), float(depth_vals.max()),
                                    value=float(depth_vals.min()), step=10.0, key="depth_single"
                                )
                    
                        # -- Time Input
                        time_vals, time_labels = try_decode_time(ds, time_var)
                        if "Time Range Avg" in plot_mode:
                            t1 = st.date_input("🕒 Start Date", value=pd.to_datetime(time_labels[0]), key="map_start")
                            t2 = st.date_input("🕒 End Date", value=pd.to_datetime(time_labels[-1]), key="map_end")
                            t1 = np.datetime64(t1)
                            t2 = np.datetime64(t2)
                        else:
                            time_sel = st.selectbox("🕒 Select Time", time_labels, key="map_single_time")
                            time_index = list(time_labels).index(time_sel)
                            raw_time_value = time_vals[time_index]

                        # ------------------ Compute time_str and depth_str ------------------ #
                        depth_str = ""
                        time_str = ""
                        
                        # Depth string
                        if "Depth Range Avg" in plot_mode:
                            depth_str = f"{dmin:.0f}–{dmax:.0f} m"
                        else:
                            depth_str = f"{selected_depth:.0f} m"
                        
                        # Time string
                        if "Time Range Avg" in plot_mode:
                            time_str = f"{pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"
                        else:
                            time_str = pd.to_datetime(raw_time_value).strftime('%Y-%m-%d')

                        
                        # ------------------ Data Extraction ------------------ #
                        data = ds[var]
                        data = data.sel({lat_var: slice(*lat_range), lon_var: slice(*lon_range)})
                    
                        if "Depth Range Avg" in plot_mode:
                            data = data.sel({depth_var: slice(dmin, dmax)})
                            data = data.mean(dim=depth_var, skipna=True)
                            depth_str = f"{dmin:.0f}–{dmax:.0f} m"
                        else:
                            data = data.sel({depth_var: selected_depth}, method="nearest")
                            depth_str = f"{selected_depth:.0f} m"
                    
                        if "Time Range Avg" in plot_mode:
                            data = data.sel({time_var: slice(t1, t2)})
                            data = data.mean(dim=time_var, skipna=True)
                            time_str = f"{pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"
                        else:
                            data = data.sel({time_var: raw_time_value})
                            time_str = pd.to_datetime(raw_time_value).strftime('%Y-%m-%d')
                    
                        # ------------------ Plotting ------------------ #
                        st.subheader("🗺️ Map View")
                        plt.rcParams['font.family'] = st.session_state.get("font_family", "DejaVu Sans")                
                    
                        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                        plot_kwargs = {
                            "ax": ax,
                            "transform": ccrs.PlateCarree(),
                            "cmap": cmap_choice,
                            "add_colorbar": True
                        }
                        if set_clim:
                            plot_kwargs["vmin"] = vmin
                            plot_kwargs["vmax"] = vmax
                    
                        im = data.squeeze().plot.pcolormesh(**plot_kwargs)
                        ax.coastlines()
                    
                        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = gl.right_labels = False
                        gl.xlabel_style = gl.ylabel_style = {'size': 12}
                    
                        if st.session_state.get("manual_ticks", False):
                            xtick_step = st.session_state.get("xtick_step")
                            ytick_step = st.session_state.get("ytick_step")
                            if xtick_step and ytick_step:
                                gl.xlocator = mticker.FixedLocator(np.arange(lon_range[0], lon_range[1] + xtick_step, xtick_step))
                                gl.ylocator = mticker.FixedLocator(np.arange(lat_range[0], lat_range[1] + ytick_step, ytick_step))
                    
                        if mask_land:
                            ax.add_feature(cfeature.LAND, facecolor=mask_color, zorder=3)
                        if mask_sea:
                            ax.add_feature(cfeature.OCEAN, facecolor=mask_color, zorder=3)
                    
                        ax.set_title(f"{plot_title}\n {time_str} | Depth: {depth_str}", fontsize=14)
                    
                        # Replace colorbar
                        if hasattr(im, 'colorbar') and im.colorbar:
                            im.colorbar.remove()
                        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.05)
                        cbar.set_label(cbar_label, fontsize=12)
                    
                        # Axis labels
                        fig_w, fig_h = fig.get_size_inches()
                        x_offset = -0.05 * (10 / fig_w)
                        y_offset = -0.1 * (6 / fig_h)
                        ax.text(0.5, y_offset, xlabel, transform=ax.transAxes, ha='center', va='top', fontsize=12)
                        ax.text(x_offset - 0.1, 0.5, ylabel, transform=ax.transAxes, ha='right', va='center', rotation='vertical', fontsize=12)
                    
                        st.pyplot(fig)

                
                        # Download
                        if save_btn:
                            buf = io.BytesIO()
                            fig.savefig(buf, format=save_format, dpi=dpi_value, bbox_inches="tight")
                            st.success(f"✅ Plot saved as {save_format.upper()} ({dpi_value} DPI)")
                            st.download_button(
                                label=f"📥 Download {save_format.upper()} file",
                                data=buf.getvalue(),
                                file_name=f"ocean_plot.{save_format}",
                                mime=f"image/{'jpeg' if save_format == 'jpg' else save_format}"
                            )
                    #---------------------------------Intercative Spatial Map View----------------------------------------------------------#

                    import plotly.graph_objects as go
                    if show_interactive_spatial_map:
                        # -- Plot Mode Selection
                        plot_mode = st.radio("🧭 Select Plot Mode", [
                            "Single Time + Single Depth",
                            "Time Range Avg + Single Depth",
                            "Single Time + Depth Range Avg",
                            "Time Range Avg + Depth Range Avg"
                        ], key="map_plot_mode")  # 🔑 Unique key is safer
                    
                        # -- Depth Input
                        depth_vals = ds[depth_var].values if depth_var else None
                        if depth_var:
                            if "Depth Range Avg" in plot_mode:
                                col1, col2 = st.columns(2)
                                with col1:
                                    dmin = st.number_input("Min Depth", float(depth_vals.min()), float(depth_vals.max()), value=0.0, key="imap_depth_min")
                                with col2:
                                    dmax = st.number_input("Max Depth", float(depth_vals.min()), float(depth_vals.max()), value=200.0, key="imap_depth_max")
                            else:
                                selected_depth = st.number_input(
                                    "Depth (m)", float(depth_vals.min()), float(depth_vals.max()),
                                    value=float(depth_vals.min()), step=10.0, key="imap_depth_single"
                                )
                    
                        # -- Time Input
                        time_vals, time_labels = try_decode_time(ds, time_var)
                        if "Time Range Avg" in plot_mode:
                            t1 = st.date_input("🕒 Start Date", value=pd.to_datetime(time_labels[0]), key="imap_start")
                            t2 = st.date_input("🕒 End Date", value=pd.to_datetime(time_labels[-1]), key="imap_end")
                            t1 = np.datetime64(t1)
                            t2 = np.datetime64(t2)
                        else:
                            time_sel = st.selectbox("🕒 Select Time", time_labels, key="imap_single_time")
                            time_index = list(time_labels).index(time_sel)
                            raw_time_value = time_vals[time_index]
                    
                        # ------------------ Data Extraction ------------------ #
                        data = ds[var]
                        data = data.sel({lat_var: slice(*lat_range), lon_var: slice(*lon_range)})
                    
                        if "Depth Range Avg" in plot_mode:
                            data = data.sel({depth_var: slice(dmin, dmax)})
                            data = data.mean(dim=depth_var, skipna=True)
                            depth_str = f"{dmin:.0f}–{dmax:.0f} m"
                        else:
                            data = data.sel({depth_var: selected_depth}, method="nearest")
                            depth_str = f"{selected_depth:.0f} m"
                    
                        if "Time Range Avg" in plot_mode:
                            data = data.sel({time_var: slice(t1, t2)})
                            data = data.mean(dim=time_var, skipna=True)
                            time_str = f"{pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"
                        else:
                            data = data.sel({time_var: raw_time_value})
                            time_str = pd.to_datetime(raw_time_value).strftime('%Y-%m-%d')
                    
                        # ------------------ Plotting ------------------ #
                        st.subheader("🎞️ Interactive Map View")
                    
                        def figsize_to_plotly(width_in, height_in, dpi=100):
                            return int(width_in * dpi), int(height_in * dpi)
                    
                        def standardize_coords(dataarray):
                            coord_map = {'latitude': None, 'longitude': None, 'time': None, 'depth': None}
                            coord_candidates = {k.lower(): k for k in dataarray.coords}
                            for standard, options in {
                                'latitude': ['lat', 'latitude'],
                                'longitude': ['lon', 'longitude'],
                                'time': ['time'],
                                'depth': ['depth', 'depth1_1', 'z']
                            }.items():
                                for opt in options:
                                    if opt in coord_candidates:
                                        coord_map[standard] = coord_candidates[opt]
                                        break
                            return coord_map
                    
                        data_2d = data.squeeze()
                        coord_map = standardize_coords(data_2d)
                    
                        lat = data_2d[coord_map['latitude']].values
                        lon = data_2d[coord_map['longitude']].values
                        z = data_2d.values
                    
                        fig = go.Figure(
                            data=go.Heatmap(
                                z=z,
                                x=lon,
                                y=lat,
                                colorscale=cmap_choice,
                                zmin=vmin if set_clim else None,
                                zmax=vmax if set_clim else None,
                                colorbar=dict(title=cbar_label),
                                hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>"
                            )
                        )
                    
                        width, height = figsize_to_plotly(10, 6)
                        fig.update_layout(
                            title=f"{plot_title}<br><sub> {time_str} | Depth: {depth_str}</sub>",
                            xaxis_title=xlabel,
                            yaxis_title=ylabel,
                            width=width,
                            height=height,
                            margin=dict(l=20, r=20, t=60, b=40)
                        )
                    
                        st.plotly_chart(fig, use_container_width=True)

                    #---------------------------------------------Spatial Map Animation--------------------------------------------------#

                    if show_time_animation:
                        import matplotlib.animation as animation
                        import io
                        import tempfile
                        import os
                    
                        st.subheader("🎞️ Time-Loop Animation (GIF)")
                        plt.rcParams['font.family'] = st.session_state.get("font_family", "DejaVu Sans")
                    
                        if time_var and time_var in ds[var].dims:
                            try:
                                # Time range selection
                                time_vals, time_labels = try_decode_time(ds, time_var)
                                time_start_default = pd.to_datetime(time_labels[0])
                                time_end_default = pd.to_datetime(time_labels[-1])
                    
                                # col1, col2 = st.columns(2)
                                # with col1:
                                #     t1 = st.date_input("🕒 Start Date", value=time_start_default)
                                # with col2:
                                #     t2 = st.date_input("🕒 End Date", value=time_end_default)

                                col1, col2 = st.columns(2)
                                with col1:
                                    t1 = st.date_input("🕒 Start Date", value=time_start_default, key="anim_start_date")
                                with col2:
                                    t2 = st.date_input("🕒 End Date", value=time_end_default, key="anim_end_date")

                                t1 = np.datetime64(t1)
                                t2 = np.datetime64(t2)
                    
                                # Plot mode: Fixed depth vs Depth-avg
                                plot_mode = st.radio("🌀 Select Animation Mode", [
                                    "Constant Depth",
                                    "Depth-averaged (Range)"
                                ])
                    
                                if plot_mode == "Constant Depth":
                                    selected_depth = st.number_input("Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=10.0)
                                else:
                                    dmin = st.number_input("Min Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=0.0)
                                    dmax = st.number_input("Max Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=200.0)
                    
                                # Select and slice data
                                da_anim = ds[var]
                                da_anim = da_anim.sel({lat_var: slice(*lat_range), lon_var: slice(*lon_range)})
                                da_anim = da_anim.sel({time_var: slice(t1, t2)})
                                time_labels = pd.to_datetime(da_anim[time_var].values)

                    
                                if plot_mode == "Constant Depth":
                                    da_anim = da_anim.sel({depth_var: selected_depth}, method="nearest")
                                else:
                                    da_anim = da_anim.sel({depth_var: slice(dmin, dmax)})
                                    da_anim = da_anim.mean(dim=depth_var, skipna=True)
                    
                                fig_anim, ax_anim = plt.subplots(figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()})
                                first_frame = da_anim.isel({time_var: 0})
                    
                                im_cbar = first_frame.plot.pcolormesh(
                                    ax=ax_anim,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap_choice,
                                    vmin=vmin if set_clim else None,
                                    vmax=vmax if set_clim else None,
                                    add_colorbar=False
                                )
                                cbar = fig_anim.colorbar(im_cbar, ax=ax_anim, orientation="vertical", shrink=0.6, pad=0.05, extend='both')
                                cbar.set_label(cbar_label, fontsize=10)
                    
                                def update_anim(frame):
                                    ax_anim.clear()
                                    frame_data = da_anim.isel({time_var: frame})
                    
                                    im = frame_data.plot.pcolormesh(
                                        ax=ax_anim,
                                        transform=ccrs.PlateCarree(),
                                        cmap=cmap_choice,
                                        vmin=vmin if set_clim else None,
                                        vmax=vmax if set_clim else None,
                                        add_colorbar=False
                                    )
                    
                                    ax_anim.coastlines()
                                    if mask_land:
                                        ax_anim.add_feature(cfeature.LAND, facecolor=mask_color, zorder=3)
                                    if mask_sea:
                                        ax_anim.add_feature(cfeature.OCEAN, facecolor=mask_color, zorder=3)
                    
                                    gl = ax_anim.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                                    gl.top_labels = gl.right_labels = False
                                    gl.xlabel_style = {'size': 10}
                                    gl.ylabel_style = {'size': 10}
                    
                                    if st.session_state.get("manual_ticks", False):
                                        xtick_step = st.session_state.get("xtick_step")
                                        ytick_step = st.session_state.get("ytick_step")
                                        if xtick_step and ytick_step:
                                            gl.xlocator = mticker.FixedLocator(np.arange(lon_range[0], lon_range[1] + xtick_step, xtick_step))
                                            gl.ylocator = mticker.FixedLocator(np.arange(lat_range[0], lat_range[1] + ytick_step, ytick_step))
                    
                                    fig_w, fig_h = fig_anim.get_size_inches()
                                    x_offset = -0.05 * (8 / fig_w)
                                    y_offset = -0.08 * (5 / fig_h)
                    
                                    lon_span = lon_range[1] - lon_range[0]
                                    lat_span = lat_range[1] - lat_range[0]
                                    label_fontsize = 8 if lon_span < 2 or lat_span < 2 else 10
                    
                                    ax_anim.text(0.5, y_offset - 0.1, xlabel, transform=ax_anim.transAxes,
                                                 ha='center', va='top', fontsize=label_fontsize)
                                    ax_anim.text(x_offset - 0.15, 0.5, ylabel, transform=ax_anim.transAxes,
                                                 ha='right', va='center', rotation='vertical', fontsize=label_fontsize)
                    
                                    # try:
                                    #     time_str = pd.to_datetime(time_labels[frame]).strftime("%Y-%m-%d")
                                    # except:
                                    #     time_str = str(time_labels[frame])[:15]
                    
                                    # title = f"{plot_title} | Time: {time_str}"

                                    try:
                                        time_str = pd.to_datetime(time_labels[frame]).strftime("%Y-%m-%d")
                                    except:
                                        time_str = str(time_labels[frame])[:15]
                                    
                                    title = f"{plot_title} | Time: {time_str}"
                                    

                                    if plot_mode == "Constant Depth":
                                        title += f" | Depth: {selected_depth} m"
                                    else:
                                        title += f" | Depth Avg: {dmin}–{dmax} m"
                    
                                    ax_anim.set_title(title, fontsize=12)
                                    return [im]
                    
                                fig_anim.subplots_adjust(left=0.2, right=1)
                    
                                ani = animation.FuncAnimation(
                                    fig_anim, update_anim, frames=da_anim.sizes[time_var], blit=False
                                )
                    
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                                    temp_gif_path = tmpfile.name
                    
                                ani.save(temp_gif_path, writer="pillow", fps=2, savefig_kwargs={'bbox_inches': 'tight'})
                    
                                with open(temp_gif_path, "rb") as f:
                                    gif_bytes = f.read()
                    
                                st.image(gif_bytes, caption="Time-animated plot", use_container_width=True)
                    
                                st.download_button(
                                    label="📥 Download GIF",
                                    data=gif_bytes,
                                    file_name=f"{var}_animation.gif",
                                    mime="image/gif"
                                )
                    
                                os.remove(temp_gif_path)
                    
                            except Exception as e:
                                st.error(f"⚠️ Failed to create animation: {e}")
                        else:
                            st.info("⏳ Animation unavailable: Time dimension not found in selected variable.")

                    
                    #-------------------------------------------- Vertical Section--------------------------------------------------#
                    
                    if show_vertical_section:
                        st.markdown("### 📉 Vertical Section Plot")

                        time_mode = st.radio("🕒 Time Mode", ["Single Time", "Time Range Average"], key="vsec_time_mode")

                        section_mode = st.selectbox("Section Mode", [
                            "Z vs Longitude (at fixed Latitude)",
                            "Z vs Latitude (at fixed Longitude)",
                            "Z vs Longitude (averaged over Latitude band)",
                            "Z vs Latitude (averaged over Longitude band)"
                        ])

                        # -- Time Input
                        time_vals, time_labels = try_decode_time(ds, time_var)

                        if time_mode == "Single Time":
                            time_sel = st.selectbox("Select Time", time_labels, key="vsec_single_time")
                            time_index = list(time_labels).index(time_sel)
                            raw_time_value = time_vals[time_index]
                        else:
                            t1 = st.date_input("Start Date", value=pd.to_datetime(time_labels[0]), key="vsec_start")
                            t2 = st.date_input("End Date", value=pd.to_datetime(time_labels[-1]), key="vsec_end")
                            t1 = np.datetime64(t1)
                            t2 = np.datetime64(t2)

                        try:
                            section = ds[var]

                            if time_var:
                                if time_mode == "Single Time":
                                    section = section.sel({time_var: raw_time_value}, method="nearest")
                                else:
                                    section = section.sel({time_var: slice(t1, t2)})
                                    section = section.mean(dim=time_var, skipna=True)


                            # if time_var and raw_time_value is not None:
                            #     section = section.sel({time_var: raw_time_value}, method="nearest")

                            # --- Depth Range Selection ---
                            depth_min = st.number_input("Min Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=0.0, step=10.0)
                            depth_max = st.number_input("Max Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=500.0, step=10.0)
                            
                            if section_mode == "Z vs Longitude (at fixed Latitude)":
                                fixed_lat = st.number_input("Fixed Latitude (°N)", float(lat_vals.min()), float(lat_vals.max()), value=15.0)
                                lon_min, lon_max = st.slider("Longitude Range (°E)", float(lon_vals.min()), float(lon_vals.max()), (50.0, 80.0))
                                section = section.sel({lat_var: fixed_lat}, method="nearest")
                                section = section.sel({lon_var: slice(lon_min, lon_max)})
                                section = section.transpose(depth_var, lon_var)
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lon_var].values
                                xlabel = "Longitude (°E)"
                                section_label = f"{fixed_lat:.2f}°N"
                    
                            elif section_mode == "Z vs Latitude (at fixed Longitude)":
                                fixed_lon = st.number_input("Fixed Longitude (°E)", float(lon_vals.min()), float(lon_vals.max()), value=60.0)
                                lat_min, lat_max = st.slider("Latitude Range (°N)", float(lat_vals.min()), float(lat_vals.max()), (0.0, 25.0))
                                section = section.sel({lon_var: fixed_lon}, method="nearest")
                                section = section.sel({lat_var: slice(lat_min, lat_max)})
                                section = section.transpose(depth_var, lat_var)  # ✅ FIX HERE
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lat_var].values
                                xlabel = "Latitude (°N)"
                                section_label = f"{fixed_lon:.2f}°E"

                    
                            elif section_mode == "Z vs Longitude (averaged over Latitude band)":
                                lat_min = st.number_input("Min Latitude", float(lat_vals.min()), float(lat_vals.max()), value=10.0)
                                lat_max = st.number_input("Max Latitude", float(lat_vals.min()), float(lat_vals.max()), value=20.0)
                                lon_min, lon_max = st.slider("Longitude Range (°E)", float(lon_vals.min()), float(lon_vals.max()), (50.0, 80.0))
                                section = section.sel({lat_var: slice(lat_min, lat_max), lon_var: slice(lon_min, lon_max)})
                                section = section.mean(dim=lat_var, skipna=True)
                                section = section.transpose(depth_var, lon_var)
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lon_var].values
                                xlabel = "Longitude (°E)"
                                section_label = f"Lat Avg ({lat_min}-{lat_max}°N)"
                    
                            elif section_mode == "Z vs Latitude (averaged over Longitude band)":
                                lon_min = st.number_input("Min Longitude", float(lon_vals.min()), float(lon_vals.max()), value=50.0)
                                lon_max = st.number_input("Max Longitude", float(lon_vals.min()), float(lon_vals.max()), value=70.0)
                                lat_min, lat_max = st.slider("Latitude Range (°N)", float(lat_vals.min()), float(lat_vals.max()), (0.0, 25.0))
                                section = section.sel({lat_var: slice(lat_min, lat_max), lon_var: slice(lon_min, lon_max)})
                                section = section.mean(dim=lon_var, skipna=True)
                                section = section.transpose(depth_var, lat_var)  # ✅ FIX HERE
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lat_var].values
                                xlabel = "Latitude (°N)"
                                section_label = f"Lon Avg ({lon_min}-{lon_max}°E)"

                    
                            else:
                                st.warning("🚫 Unknown section mode selected.")
                                st.stop()
                    
                            z_vals = section[depth_var].values
                            data_vals = section.values
                    
                            # --- Plot ---
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cf = ax.contourf(
                                x_vals, z_vals, data_vals,
                                levels=50, cmap=cmap_choice,
                                vmin=vmin if set_clim else None,
                                vmax=vmax if set_clim else None
                            )
                            ax.invert_yaxis()
                            # ax.set_title(f"{var} Vertical Section at {section_label}", fontsize=14)
                            if time_mode == "Single Time":
                                time_str = pd.to_datetime(raw_time_value).strftime('%Y-%m-%d')
                            else:
                                time_str = f"{pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"

                            ax.set_title(f"{var} Vertical Section at {section_label}\n {time_str}", fontsize=14)
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel("Depth (m)")
                            cbar = fig.colorbar(cf, ax=ax)
                            cbar.set_label(cbar_label)
                            st.pyplot(fig)
                    
                            # --- Save Option ---
                            if save_btn:
                                buf = io.BytesIO()
                                fig.savefig(buf, format=save_format, dpi=dpi_value, bbox_inches="tight")
                                st.download_button(
                                    label=f"📥 Download {save_format.upper()}",
                                    data=buf.getvalue(),
                                    file_name=f"{var}_section_{section_label}.{save_format}",
                                    mime=f"image/{'jpeg' if save_format == 'jpg' else save_format}"
                                )
                    
                        except Exception as e:
                            st.error(f"❌ Failed to plot vertical section: {e}")

                    #----------------------------------- Interactive Vertical Section--------------------------------------------------#
                  
                    if show_interactive_vertical_section:
                        st.markdown("### 🧪 Interactive Vertical Section")
                    
                        # -- Time Mode Selector --
                        time_mode = st.radio("🕒 Time Mode (Interactive)", ["Single Time", "Time Range Average"], key="vsec_time_mode_inter")
                    
                        section_mode = st.selectbox("Section Mode (Interactive)", [
                            "Z vs Longitude (at fixed Latitude)",
                            "Z vs Latitude (at fixed Longitude)",
                            "Z vs Longitude (averaged over Latitude band)",
                            "Z vs Latitude (averaged over Longitude band)"
                        ])
                    
                        # -- Time Input
                        time_vals, time_labels = try_decode_time(ds, time_var)
                        if time_mode == "Single Time":
                            time_sel = st.selectbox("Select Time", time_labels, key="vsec_single_time_inter")
                            time_index = list(time_labels).index(time_sel)
                            raw_time_value = time_vals[time_index]
                        else:
                            t1 = st.date_input("Start Date", value=pd.to_datetime(time_labels[0]), key="vsec_start_inter")
                            t2 = st.date_input("End Date", value=pd.to_datetime(time_labels[-1]), key="vsec_end_inter")
                            t1 = np.datetime64(t1)
                            t2 = np.datetime64(t2)
                    
                        try:
                            section = ds[var]
                    
                            if time_var:
                                if time_mode == "Single Time":
                                    section = section.sel({time_var: raw_time_value}, method="nearest")
                                else:
                                    section = section.sel({time_var: slice(t1, t2)})
                                    section = section.mean(dim=time_var, skipna=True)
                    
                            # --- Depth Range Selection ---
                            depth_min = st.number_input("Min Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=0.0, step=10.0, key="dmin_int")
                            depth_max = st.number_input("Max Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=500.0, step=10.0, key="dmax_int")
                    
                            if section_mode == "Z vs Longitude (at fixed Latitude)":
                                fixed_lat = st.number_input("Fixed Latitude (°N)", float(lat_vals.min()), float(lat_vals.max()), value=15.0, key="fixed_lat_int")
                                lon_min, lon_max = st.slider("Longitude Range (°E)", float(lon_vals.min()), float(lon_vals.max()), (50.0, 80.0), key="lon_range_int")
                                section = section.sel({lat_var: fixed_lat}, method="nearest")
                                section = section.sel({lon_var: slice(lon_min, lon_max)})
                                section = section.transpose(depth_var, lon_var)
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lon_var].values
                                xlabel = "Longitude (°E)"
                                section_label = f"{fixed_lat:.2f}°N"
                    
                            elif section_mode == "Z vs Latitude (at fixed Longitude)":
                                fixed_lon = st.number_input("Fixed Longitude (°E)", float(lon_vals.min()), float(lon_vals.max()), value=60.0, key="fixed_lon_int")
                                lat_min, lat_max = st.slider("Latitude Range (°N)", float(lat_vals.min()), float(lat_vals.max()), (0.0, 25.0), key="lat_range_int")
                                section = section.sel({lon_var: fixed_lon}, method="nearest")
                                section = section.sel({lat_var: slice(lat_min, lat_max)})
                                section = section.transpose(depth_var, lat_var)
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lat_var].values
                                xlabel = "Latitude (°N)"
                                section_label = f"{fixed_lon:.2f}°E"
                    
                            elif section_mode == "Z vs Longitude (averaged over Latitude band)":
                                lat_min = st.number_input("Min Latitude", float(lat_vals.min()), float(lat_vals.max()), value=10.0, key="latmin_int")
                                lat_max = st.number_input("Max Latitude", float(lat_vals.min()), float(lat_vals.max()), value=20.0, key="latmax_int")
                                lon_min, lon_max = st.slider("Longitude Range (°E)", float(lon_vals.min()), float(lon_vals.max()), (50.0, 80.0), key="lon_avg_int")
                                section = section.sel({lat_var: slice(lat_min, lat_max), lon_var: slice(lon_min, lon_max)})
                                section = section.mean(dim=lat_var, skipna=True)
                                section = section.transpose(depth_var, lon_var)
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lon_var].values
                                xlabel = "Longitude (°E)"
                                section_label = f"Lat Avg ({lat_min}-{lat_max}°N)"
                    
                            elif section_mode == "Z vs Latitude (averaged over Longitude band)":
                                lon_min = st.number_input("Min Longitude", float(lon_vals.min()), float(lon_vals.max()), value=50.0, key="lonmin_int")
                                lon_max = st.number_input("Max Longitude", float(lon_vals.min()), float(lon_vals.max()), value=70.0, key="lonmax_int")
                                lat_min, lat_max = st.slider("Latitude Range (°N)", float(lat_vals.min()), float(lat_vals.max()), (0.0, 25.0), key="lat_avg_int")
                                section = section.sel({lat_var: slice(lat_min, lat_max), lon_var: slice(lon_min, lon_max)})
                                section = section.mean(dim=lon_var, skipna=True)
                                section = section.transpose(depth_var, lat_var)
                                section = section.sel({depth_var: slice(depth_min, depth_max)})
                                x_vals = section[lat_var].values
                                xlabel = "Latitude (°N)"
                                section_label = f"Lon Avg ({lon_min}-{lon_max}°E)"
                    
                            else:
                                st.warning("🚫 Unknown section mode selected.")
                                st.stop()
                    
                            z_vals = section[depth_var].values
                            data_vals = section.values
                    
                            import plotly.graph_objects as go
                    
                            if time_mode == "Single Time":
                                time_str = pd.to_datetime(raw_time_value).strftime('%Y-%m-%d')
                            else:
                                time_str = f"{pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"
                    
                            fig_plotly = go.Figure(data=go.Heatmap(
                                z=data_vals,
                                x=x_vals,
                                y=z_vals,
                                colorscale=cmap_choice,
                                zmin=vmin if set_clim else None,
                                zmax=vmax if set_clim else None,
                                colorbar=dict(title=cbar_label),
                                hovertemplate=(
                                    f"{xlabel}: %{{x:.2f}}<br>"
                                    "Depth: %{y:.1f} m<br>"
                                    f"{var}: %{{z:.2f}}<extra></extra>"
                                )
                            ))
                    
                            fig_plotly.update_layout(
                                title=f"{var} Vertical Section at {section_label}<br><sub>{time_str}</sub>",
                                xaxis_title=xlabel,
                                yaxis_title="Depth (m)",
                                yaxis_autorange="reversed",
                                width=900,
                                height=600
                            )
                    
                            st.plotly_chart(fig_plotly, use_container_width=True)
                    
                        except Exception as e:
                            st.error(f"❌ Failed to plot interactive vertical section: {e}")

                    #---------------------------------Timeseries View----------------------------------------------------------#
                    if show_timeseries_plot:
                        ts_mode = st.selectbox("📌 Select Time Series Mode", [
                            "Point • Single Depth",
                            "Point • Depth Range Average",
                            "Point • Full Depth Average",
                            "Grid • Single Depth",
                            "Grid • Depth Range Average",
                            "Grid • Full Depth Average"
                        ])
                    
                        # === Shared Input: Variable Selection ===
                        # var = st.selectbox("Select Variable", list(ds.data_vars))
                        da = ds[var]
                    
                        # === Coordinate Map ===
                        coord_map = detect_coord_names(ds)
                        lat_var, lon_var, depth_var, time_var = coord_map['latitude'], coord_map['longitude'], coord_map['depth'], coord_map['time']
                    
                        time_vals, time_labels = try_decode_time(ds, time_var)
                        da.coords[time_var] = time_labels
                    
                        # === UI Logic ===
                        if "Point" in ts_mode:
                            lat_pt = st.number_input("Latitude (°N)", float(ds[lat_var].min()), float(ds[lat_var].max()), value=15.0)
                            lon_pt = st.number_input("Longitude (°E)", float(ds[lon_var].min()), float(ds[lon_var].max()), value=60.0)
                    
                        if "Grid" in ts_mode:
                            col1, col2 = st.columns(2)
                            with col1:
                                lat_min = st.number_input("Min Latitude", float(ds[lat_var].min()), float(ds[lat_var].max()), value=10.0)
                                lat_max = st.number_input("Max Latitude", float(ds[lat_var].min()), float(ds[lat_var].max()), value=20.0)
                            with col2:
                                lon_min = st.number_input("Min Longitude", float(ds[lon_var].min()), float(ds[lon_var].max()), value=50.0)
                                lon_max = st.number_input("Max Longitude", float(ds[lon_var].min()), float(ds[lon_var].max()), value=70.0)
                    
                        if "Single Depth" in ts_mode:
                            depth_val = st.number_input("Select Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=float(ds[depth_var].min()))
                        elif "Depth Range" in ts_mode:
                            depth_min = st.number_input("Min Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=0.0)
                            depth_max = st.number_input("Max Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=200.0)
                    
                        # === Subsetting and Averaging ===
                        try:
                            if "Point" in ts_mode:
                                da_sel = da.sel({lat_var: lat_pt, lon_var: lon_pt}, method="nearest")
                            elif "Grid" in ts_mode:
                                da_sel = da.sel({lat_var: slice(lat_min, lat_max), lon_var: slice(lon_min, lon_max)})
                                da_sel = da_sel.mean(dim=[lat_var, lon_var], skipna=True)
                    
                            if "Single Depth" in ts_mode:
                                da_sel = da_sel.sel({depth_var: depth_val}, method="nearest")
                            elif "Depth Range" in ts_mode:
                                da_sel = da_sel.sel({depth_var: slice(depth_min, depth_max)})
                                da_sel = da_sel.mean(dim=depth_var, skipna=True)
                            elif "Full Depth" in ts_mode:
                                da_sel = da_sel.mean(dim=depth_var, skipna=True)
                    
                            # === Plotting ===
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(da_sel[time_var].values, da_sel.values, marker='o')
                            ax.set_title(f"{var} Time Series")
                            ax.set_xlabel("Time")
                            ax.set_ylabel(var)
                            ax.grid(True)
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    
                            if save_btn:
                                buf = io.BytesIO()
                                fig.savefig(buf, format=save_format, dpi=dpi_value, bbox_inches="tight")
                                st.download_button(
                                    label=f"📥 Download {save_format.upper()}",
                                    data=buf.getvalue(),
                                    file_name=f"{var}_timeseries_{ts_mode.replace(' ', '_')}.{save_format}",
                                    mime=f"image/{'jpeg' if save_format == 'jpg' else save_format}"
                                )
                    
                        except Exception as e:
                            st.error(f"❌ Failed to plot time series: {e}")
                    
                    #---------------------------Normal Vertical Profile ----------------------------------#
                    
                    if show_vertical_profile:
                        st.markdown("### 📉 Vertical Profile")
                    
                        coord_map = detect_coord_names(ds_sel)
                        lat_key = coord_map["latitude"]
                        lon_key = coord_map["longitude"]
                        depth_key = coord_map["depth"]
                        time_key = coord_map["time"]
                    
                        profile_mode = st.selectbox("Profile Mode", [
                            "Single Point (lat, lon)",
                            "Lat-Lon Box Averaged",
                            "Latitudinal Transect (fixed lon)",
                            "Longitudinal Transect (fixed lat)"
                        ])
                    
                        time_profile_mode = st.radio("Time Aggregation Mode", [
                            "Use selected time only",
                            "Average over selected time range",
                            "Plot all times"
                        ])
                    
                        if not all([lat_key, lon_key, depth_key]):
                            st.error("❌ Could not detect necessary coordinate names (lat/lon/depth).")
                        else:
                            try:
                                depth_min = st.number_input("Min Depth (m)", float(ds[depth_key].min()), float(ds[depth_key].max()), value=0.0)
                                depth_max = st.number_input("Max Depth (m)", float(ds[depth_key].min()), float(ds[depth_key].max()), value=500.0)
                    
                                if profile_mode == "Single Point (lat, lon)":
                                    input_lat = st.number_input("Latitude (°N)", float(ds[lat_key].min()), float(ds[lat_key].max()), value=15.0)
                                    input_lon = st.number_input("Longitude (°E)", float(ds[lon_key].min()), float(ds[lon_key].max()), value=60.0)
                                    profile = ds[var].sel({lat_key: input_lat, lon_key: input_lon}, method="nearest")
                                    label = f"({input_lat:.2f}°N, {input_lon:.2f}°E)"
                    
                                elif profile_mode == "Lat-Lon Box Averaged":
                                    lat_min = st.number_input("Min Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=10.0)
                                    lat_max = st.number_input("Max Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=20.0)
                                    lon_min = st.number_input("Min Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=50.0)
                                    lon_max = st.number_input("Max Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=70.0)
                                    profile = ds[var].sel({lat_key: slice(lat_min, lat_max), lon_key: slice(lon_min, lon_max)})
                                    profile = profile.mean(dim=[lat_key, lon_key], skipna=True)
                                    label = f"Grid Avg ({lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E)"
                    
                                elif profile_mode == "Latitudinal Transect (fixed lon)":
                                    lon_fixed = st.number_input("Fixed Longitude (°E)", float(ds[lon_key].min()), float(ds[lon_key].max()), value=60.0)
                                    lat_min = st.number_input("Min Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=10.0)
                                    lat_max = st.number_input("Max Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=20.0)
                                    profile = ds[var].sel({lon_key: lon_fixed}, method="nearest")
                                    profile = profile.sel({lat_key: slice(lat_min, lat_max)})
                                    profile = profile.mean(dim=lat_key, skipna=True)
                                    label = f"Lat Avg ({lat_min}-{lat_max}°N) at {lon_fixed}°E"
                    
                                elif profile_mode == "Longitudinal Transect (fixed lat)":
                                    lat_fixed = st.number_input("Fixed Latitude (°N)", float(ds[lat_key].min()), float(ds[lat_key].max()), value=15.0)
                                    lon_min = st.number_input("Min Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=50.0)
                                    lon_max = st.number_input("Max Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=70.0)
                                    profile = ds[var].sel({lat_key: lat_fixed}, method="nearest")
                                    profile = profile.sel({lon_key: slice(lon_min, lon_max)})
                                    profile = profile.mean(dim=lon_key, skipna=True)
                                    label = f"Lon Avg ({lon_min}-{lon_max}°E) at {lat_fixed}°N"
                    
                                if time_key and time_key in profile.dims:
                                    time_vals, time_labels = try_decode_time(ds, time_key)

                                    if time_profile_mode == "Use selected time only":
                                        time_vals, time_labels = try_decode_time(ds, time_key)
                                        time_sel_label = st.selectbox("Select Time", time_labels, key="vprof_single_timee")
                                        time_index = list(time_labels).index(time_sel_label)
                                        time_sel = time_vals[time_index]
                                        profile = profile.sel({time_key: time_sel}, method="nearest")
                                    
                                    elif time_profile_mode == "Average over selected time range":
                                        time_vals, time_labels = try_decode_time(ds, time_key)
                                        t1 = st.date_input("Start Date", value=pd.to_datetime(time_labels[0]), key="vprof_t1")
                                        t2 = st.date_input("End Date", value=pd.to_datetime(time_labels[-1]), key="vprof_t2")
                                        t1 = np.datetime64(t1)
                                        t2 = np.datetime64(t2)
                                        profile = profile.sel({time_key: slice(t1, t2)}).mean(dim=time_key, skipna=True)
                                    
                                    elif time_profile_mode == "Plot all times":
                                        pass  # keep time dimension
                    
                                profile = profile.sel({depth_key: slice(depth_min, depth_max)})
                    
                                if depth_key in profile.coords:
                                    depth_vals = profile[depth_key].values
                                elif depth_key in ds.coords:
                                    depth_vals = ds[depth_key].values
                                else:
                                    st.error("❌ Could not find depth values in dataset.")
                                    st.stop()
                    
                                fig, ax = plt.subplots(figsize=(6, 5))
                    
                                if time_profile_mode == "Plot all times" and time_key in profile.dims:
                                    for t in range(profile.sizes[time_key]):
                                        time_val = profile[time_key][t].values
                                        ax.plot(profile.isel({time_key: t}).values, depth_vals, marker='o', linestyle='-', label=pd.to_datetime(time_val).strftime('%Y-%m-%d'))
                                    ax.legend(fontsize=8)
                                else:
                                    var_vals = profile.values
                                    ax.plot(var_vals, depth_vals, marker='o', linestyle='-')
                    
                                ax.set_xlabel(var)
                                ax.set_ylabel("Depth (m)")
                                ax.invert_yaxis()
                                # ax.set_title(f"{var} Profile at {label}")
                                # Format time label
                                if time_profile_mode == "Use selected time only":
                                    time_title = f"\n {pd.to_datetime(time_sel).strftime('%Y-%m-%d')}"
                                elif time_profile_mode == "Average over selected time range":
                                    time_title = f"\n {pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"
                                elif time_profile_mode == "Plot all times":
                                    time_title = "\n All Available Times"
                                
                                # Final title
                                ax.set_title(f"{var} Vertical Profile\n{label}{time_title}")

                                st.pyplot(fig)
                    
                            except Exception as e:
                                st.error(f"❌ Failed to plot vertical profile: {e}")

                    
                                if save_btn:
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format=save_format, dpi=dpi_value, bbox_inches="tight")
                                    st.success(f"✅ Plot saved as {save_format.upper()} ({dpi_value} DPI)")
                                    st.download_button(
                                        label=f"📥 Download {save_format.upper()} file",
                                        data=buf.getvalue(),
                                        file_name=f"ocean_plot.{save_format}",
                                        mime=f"image/{'jpeg' if save_format == 'jpg' else save_format}"
                                    )
                    
                            except Exception as e:
                                st.error(f"❌ Failed to extract profile: {e}")

                    # -----------------------------------Interactive Vetical Profile ---------------------------------------------------#
                    
                    # import plotly.graph_objects as go
                    # if show_interactive_vertical_profile:
                    #     st.markdown("### 📉 Interactive Vertical Profile")
                    
                    #     coord_map = detect_coord_names(ds_sel)
                    #     lat_key = coord_map["latitude"]
                    #     lon_key = coord_map["longitude"]
                    #     depth_key = coord_map["depth"]
                    #     time_key = coord_map["time"]
                    
                    #     profile_mode = st.selectbox("Profile Mode", [
                    #         "Single Point (lat, lon)",
                    #         "Lat-Lon Box Averaged",
                    #         "Latitudinal Transect (fixed lon)",
                    #         "Longitudinal Transect (fixed lat)"
                    #     ])
                    
                    #     time_profile_mode = st.radio("Time Aggregation Mode", [
                    #         "Use selected time only",
                    #         "Average over selected time range",
                    #         "Plot all times"
                    #     ])
                    
                    #     if not all([lat_key, lon_key, depth_key]):
                    #         st.error("❌ Could not detect necessary coordinate names (lat/lon/depth).")
                    #     else:
                    #         try:
                    #             depth_min = st.number_input("Min Depth (m)", float(ds[depth_key].min()), float(ds[depth_key].max()), value=0.0)
                    #             depth_max = st.number_input("Max Depth (m)", float(ds[depth_key].min()), float(ds[depth_key].max()), value=500.0)
                    
                    #             if profile_mode == "Single Point (lat, lon)":
                    #                 input_lat = st.number_input("Latitude (°N)", float(ds[lat_key].min()), float(ds[lat_key].max()), value=15.0)
                    #                 input_lon = st.number_input("Longitude (°E)", float(ds[lon_key].min()), float(ds[lon_key].max()), value=60.0)
                    #                 profile = ds[var].sel({lat_key: input_lat, lon_key: input_lon}, method="nearest")
                    #                 label = f"({input_lat:.2f}°N, {input_lon:.2f}°E)"
                    
                    #             elif profile_mode == "Lat-Lon Box Averaged":
                    #                 lat_min = st.number_input("Min Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=10.0)
                    #                 lat_max = st.number_input("Max Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=20.0)
                    #                 lon_min = st.number_input("Min Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=50.0)
                    #                 lon_max = st.number_input("Max Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=70.0)
                    #                 profile = ds[var].sel({lat_key: slice(lat_min, lat_max), lon_key: slice(lon_min, lon_max)})
                    #                 profile = profile.mean(dim=[lat_key, lon_key], skipna=True)
                    #                 label = f"Grid Avg ({lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E)"
                    
                    #             elif profile_mode == "Latitudinal Transect (fixed lon)":
                    #                 lon_fixed = st.number_input("Fixed Longitude (°E)", float(ds[lon_key].min()), float(ds[lon_key].max()), value=60.0)
                    #                 lat_min = st.number_input("Min Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=10.0)
                    #                 lat_max = st.number_input("Max Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=20.0)
                    #                 profile = ds[var].sel({lon_key: lon_fixed}, method="nearest")
                    #                 profile = profile.sel({lat_key: slice(lat_min, lat_max)})
                    #                 profile = profile.mean(dim=lat_key, skipna=True)
                    #                 label = f"Lat Avg ({lat_min}-{lat_max}°N) at {lon_fixed}°E"
                    
                    #             elif profile_mode == "Longitudinal Transect (fixed lat)":
                    #                 lat_fixed = st.number_input("Fixed Latitude (°N)", float(ds[lat_key].min()), float(ds[lat_key].max()), value=15.0)
                    #                 lon_min = st.number_input("Min Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=50.0)
                    #                 lon_max = st.number_input("Max Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=70.0)
                    #                 profile = ds[var].sel({lat_key: lat_fixed}, method="nearest")
                    #                 profile = profile.sel({lon_key: slice(lon_min, lon_max)})
                    #                 profile = profile.mean(dim=lon_key, skipna=True)
                    #                 label = f"Lon Avg ({lon_min}-{lon_max}°E) at {lat_fixed}°N"
                    
                    #             if time_key and time_key in profile.dims:
                    #                 if time_profile_mode == "Use selected time only":
                    #                     profile = profile.sel({time_key: time_sel}, method="nearest")
                    #                 elif time_profile_mode == "Average over selected time range":
                    #                     time_min = st.number_input("Min Time Index", 0, int(ds.dims[time_key]) - 1, value=0)
                    #                     time_max = st.number_input("Max Time Index", time_min, int(ds.dims[time_key]) - 1, value=5)
                    #                     profile = profile.isel({time_key: slice(time_min, time_max + 1)}).mean(dim=time_key, skipna=True)
                    #                 elif time_profile_mode == "Plot all times":
                    #                     pass
                    
                    #             profile = profile.sel({depth_key: slice(depth_min, depth_max)})
                    
                    #             if depth_key in profile.coords:
                    #                 depth_vals = profile[depth_key].values
                    #             elif depth_key in ds.coords:
                    #                 depth_vals = ds[depth_key].values
                    #             else:
                    #                 st.error("❌ Could not find depth values in dataset.")
                    #                 st.stop()
                    
                    #             import plotly.graph_objects as go
                    #             fig = go.Figure()
                    
                    #             if time_profile_mode == "Plot all times" and time_key in profile.dims:
                    #                 for t in range(profile.sizes[time_key]):
                    #                     time_val = profile[time_key][t].values
                    #                     fig.add_trace(go.Scatter(
                    #                         y=depth_vals,
                    #                         x=profile.isel({time_key: t}).values,
                    #                         mode='lines+markers',
                    #                         name=f"t={time_val}"
                    #                     ))
                    #             else:
                    #                 fig.add_trace(go.Scatter(
                    #                     y=depth_vals,
                    #                     x=profile.values,
                    #                     mode='lines+markers',
                    #                     name=var
                    #                 ))
                    
                    #             fig.update_layout(
                    #                 title=f"{var} Profile at {label}",
                    #                 xaxis_title=var,
                    #                 yaxis_title="Depth (m)",
                    #                 yaxis_autorange="reversed",
                    #                 height=500,
                    #                 width=500
                    #             )
                    #             st.plotly_chart(fig)
                    
                    #         except Exception as e:
                    #             st.error(f"❌ Failed to extract profile: {e}")

                    import plotly.graph_objects as go
                    if show_interactive_vertical_profile:
                        st.markdown("### 📉 Interactive Vertical Profile")
                    
                        coord_map = detect_coord_names(ds_sel)
                        lat_key = coord_map["latitude"]
                        lon_key = coord_map["longitude"]
                        depth_key = coord_map["depth"]
                        time_key = coord_map["time"]
                    
                        profile_mode = st.selectbox("Profile Mode", [
                            "Single Point (lat, lon)",
                            "Lat-Lon Box Averaged",
                            "Latitudinal Transect (fixed lon)",
                            "Longitudinal Transect (fixed lat)"
                        ])
                    
                        time_profile_mode = st.radio("Time Aggregation Mode", [
                            "Use selected time only",
                            "Average over selected time range",
                            "Plot all times"
                        ])
                    
                        if not all([lat_key, lon_key, depth_key]):
                            st.error("❌ Could not detect necessary coordinate names (lat/lon/depth).")
                        else:
                            try:
                                depth_min = st.number_input("Min Depth (m)", float(ds[depth_key].min()), float(ds[depth_key].max()), value=0.0)
                                depth_max = st.number_input("Max Depth (m)", float(ds[depth_key].min()), float(ds[depth_key].max()), value=500.0)
                    
                                if profile_mode == "Single Point (lat, lon)":
                                    input_lat = st.number_input("Latitude (°N)", float(ds[lat_key].min()), float(ds[lat_key].max()), value=15.0)
                                    input_lon = st.number_input("Longitude (°E)", float(ds[lon_key].min()), float(ds[lon_key].max()), value=60.0)
                                    profile = ds[var].sel({lat_key: input_lat, lon_key: input_lon}, method="nearest")
                                    label = f"({input_lat:.2f}°N, {input_lon:.2f}°E)"
                    
                                elif profile_mode == "Lat-Lon Box Averaged":
                                    lat_min = st.number_input("Min Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=10.0)
                                    lat_max = st.number_input("Max Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=20.0)
                                    lon_min = st.number_input("Min Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=50.0)
                                    lon_max = st.number_input("Max Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=70.0)
                                    profile = ds[var].sel({lat_key: slice(lat_min, lat_max), lon_key: slice(lon_min, lon_max)})
                                    profile = profile.mean(dim=[lat_key, lon_key], skipna=True)
                                    label = f"Grid Avg ({lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E)"
                    
                                elif profile_mode == "Latitudinal Transect (fixed lon)":
                                    lon_fixed = st.number_input("Fixed Longitude (°E)", float(ds[lon_key].min()), float(ds[lon_key].max()), value=60.0)
                                    lat_min = st.number_input("Min Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=10.0)
                                    lat_max = st.number_input("Max Latitude", float(ds[lat_key].min()), float(ds[lat_key].max()), value=20.0)
                                    profile = ds[var].sel({lon_key: lon_fixed}, method="nearest")
                                    profile = profile.sel({lat_key: slice(lat_min, lat_max)})
                                    profile = profile.mean(dim=lat_key, skipna=True)
                                    label = f"Lat Avg ({lat_min}-{lat_max}°N) at {lon_fixed}°E"
                    
                                elif profile_mode == "Longitudinal Transect (fixed lat)":
                                    lat_fixed = st.number_input("Fixed Latitude (°N)", float(ds[lat_key].min()), float(ds[lat_key].max()), value=15.0)
                                    lon_min = st.number_input("Min Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=50.0)
                                    lon_max = st.number_input("Max Longitude", float(ds[lon_key].min()), float(ds[lon_key].max()), value=70.0)
                                    profile = ds[var].sel({lat_key: lat_fixed}, method="nearest")
                                    profile = profile.sel({lon_key: slice(lon_min, lon_max)})
                                    profile = profile.mean(dim=lon_key, skipna=True)
                                    label = f"Lon Avg ({lon_min}-{lon_max}°E) at {lat_fixed}°N"
                    
                                # Handle time
                                if time_key and time_key in profile.dims:
                                    if time_profile_mode == "Use selected time only":
                                        time_vals, time_labels = try_decode_time(ds, time_key)
                                        time_sel_str = st.selectbox("Select Time", time_labels, key="prof_time_sel")
                                        time_index = list(time_labels).index(time_sel_str)
                                        time_sel = time_vals[time_index]
                                        profile = profile.sel({time_key: time_sel}, method="nearest")
                                        time_title = f"\n {pd.to_datetime(time_sel).strftime('%Y-%m-%d')}"
                    
                                    elif time_profile_mode == "Average over selected time range":
                                        time_vals, time_labels = try_decode_time(ds, time_key)
                                        time_min_idx = st.number_input("Min Time Index", 0, int(ds.dims[time_key]) - 1, value=0)
                                        time_max_idx = st.number_input("Max Time Index", time_min_idx, int(ds.dims[time_key]) - 1, value=5)
                                        t1 = time_vals[time_min_idx]
                                        t2 = time_vals[time_max_idx]
                                        profile = profile.isel({time_key: slice(time_min_idx, time_max_idx + 1)}).mean(dim=time_key, skipna=True)
                                        time_title = f"\n {pd.to_datetime(t1).strftime('%Y-%m-%d')} to {pd.to_datetime(t2).strftime('%Y-%m-%d')}"
                    
                                    elif time_profile_mode == "Plot all times":
                                        time_title = "\n🕒 All Available Times"
                                    else:
                                        time_title = ""
                                else:
                                    time_title = ""
                    
                                profile = profile.sel({depth_key: slice(depth_min, depth_max)})
                    
                                if depth_key in profile.coords:
                                    depth_vals = profile[depth_key].values
                                elif depth_key in ds.coords:
                                    depth_vals = ds[depth_key].values
                                else:
                                    st.error("❌ Could not find depth values in dataset.")
                                    st.stop()
                    
                                fig = go.Figure()
                    
                                if time_profile_mode == "Plot all times" and time_key in profile.dims:
                                    for t in range(profile.sizes[time_key]):
                                        time_val = profile[time_key][t].values
                                        fig.add_trace(go.Scatter(
                                            y=depth_vals,
                                            x=profile.isel({time_key: t}).values,
                                            mode='lines+markers',
                                            name=f"t={pd.to_datetime(time_val).strftime('%Y-%m-%d')}"
                                        ))
                                else:
                                    fig.add_trace(go.Scatter(
                                        y=depth_vals,
                                        x=profile.values,
                                        mode='lines+markers',
                                        name=var
                                    ))
                    
                                fig.update_layout(
                                    title=f"{var} Vertical Profile<br>{label}{time_title}",
                                    xaxis_title=var,
                                    yaxis_title="Depth (m)",
                                    yaxis_autorange="reversed",
                                    height=500,
                                    width=500
                                )
                                st.plotly_chart(fig)
                    
                            except Exception as e:
                                st.error(f"❌ Failed to extract profile: {e}")

                        
                    #---------------------------------------- Hovmoller ----------------------------------------------------#
                    
                    if show_hovmoller:
                        hov_mode = st.selectbox("📌 Select Hovmöller Mode", [
                            "Longitude vs Time • Fixed Lat & Depth",
                            "Longitude vs Time • Fixed Lat & Depth-avg",
                            "Latitude vs Time • Fixed Lon & Depth",
                            "Latitude vs Time • Fixed Lon & Depth-avg",
                            "Depth vs Time • Fixed Lat & Lon",
                            "Depth vs Time • Grid Avg (Lat-Lon box)"
                        ])
                    
                        da = ds[var]
                    
                        coord_map = detect_coord_names(ds)
                        lat_var, lon_var, depth_var, time_var = coord_map['latitude'], coord_map['longitude'], coord_map['depth'], coord_map['time']
                    
                        # --- Decode time and enable range selection ---
                        time_vals, time_labels = try_decode_time(ds, time_var)
                        da.coords[time_var] = time_labels
                    
                        time_start_default = pd.to_datetime(time_labels[0])
                        time_end_default = pd.to_datetime(time_labels[-1])
                    
                        col1, col2 = st.columns(2)
                        with col1:
                            # t1 = st.date_input("🕒 Start Date", value=time_start_default)
                            t1 = st.date_input("🕒 Start Date", value=time_start_default, key="hov_start_date")
                        with col2:
                            # t2 = st.date_input("🕒 End Date", value=time_end_default)
                            t2 = st.date_input("🕒 End Date", value=time_end_default, key="hov_end_date")
                    
                        t1 = np.datetime64(t1)
                        t2 = np.datetime64(t2)
                    
                        try:
                            if hov_mode.startswith("Longitude"):
                                fixed_lat = st.number_input("Latitude (°N)", float(ds[lat_var].min()), float(ds[lat_var].max()), value=15.0)
                                lon_min = st.number_input("Min Longitude", float(ds[lon_var].min()), float(ds[lon_var].max()), value=float(ds[lon_var].min()))
                                lon_max = st.number_input("Max Longitude", float(ds[lon_var].min()), float(ds[lon_var].max()), value=float(ds[lon_var].max()))
                    
                                if "Depth-avg" in hov_mode:
                                    d1 = st.number_input("Min Depth", float(ds[depth_var].min()), float(ds[depth_var].max()), value=0.0)
                                    d2 = st.number_input("Max Depth", float(ds[depth_var].min()), float(ds[depth_var].max()), value=200.0)
                                    da_sel = da.sel({lon_var: slice(lon_min, lon_max), depth_var: slice(d1, d2)})
                                    da_sel = da_sel.sel({lat_var: fixed_lat}, method="nearest").mean(dim=depth_var, skipna=True)
                                else:
                                    fixed_depth = st.number_input("Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=10.0, key="hov_depth")

                                    da_sel = da.sel({lon_var: slice(lon_min, lon_max), depth_var: fixed_depth})
                                    da_sel = da_sel.sel({lat_var: fixed_lat}, method="nearest")
                    
                                da_sel = da_sel.sel({time_var: slice(t1, t2)})
                                hov_x = da_sel[lon_var]
                                hov_y = da_sel[time_var]
                                hov_z = da_sel.transpose(time_var, lon_var)
                    
                            elif hov_mode.startswith("Latitude"):
                                fixed_lon = st.number_input("Longitude (°E)", float(ds[lon_var].min()), float(ds[lon_var].max()), value=60.0, key="hov_depth")
                                lat_min = st.number_input("Min Latitude", float(ds[lat_var].min()), float(ds[lat_var].max()), value=float(ds[lat_var].min()))
                                lat_max = st.number_input("Max Latitude", float(ds[lat_var].min()), float(ds[lat_var].max()), value=float(ds[lat_var].max()))
                    
                                if "Depth-avg" in hov_mode:
                                    d1 = st.number_input("Min Depth", float(ds[depth_var].min()), float(ds[depth_var].max()), value=0.0)
                                    d2 = st.number_input("Max Depth", float(ds[depth_var].min()), float(ds[depth_var].max()), value=200.0)
                                    da_sel = da.sel({lat_var: slice(lat_min, lat_max), depth_var: slice(d1, d2)})
                                    da_sel = da_sel.sel({lon_var: fixed_lon}, method="nearest").mean(dim=depth_var, skipna=True)
                                else:
                                    fixed_depth = st.number_input("Depth (m)", float(ds[depth_var].min()), float(ds[depth_var].max()), value=10.0, key="hov_depth")
                                    da_sel = da.sel({lat_var: slice(lat_min, lat_max), depth_var: fixed_depth})
                                    da_sel = da_sel.sel({lon_var: fixed_lon}, method="nearest")
                    
                                da_sel = da_sel.sel({time_var: slice(t1, t2)})
                                hov_x = da_sel[lat_var]
                                hov_y = da_sel[time_var]
                                hov_z = da_sel.transpose(time_var, lat_var)
                    
                            elif hov_mode == "Depth vs Time • Fixed Lat & Lon":
                                lat_pt = st.number_input("Latitude (°N)", float(ds[lat_var].min()), float(ds[lat_var].max()), value=15.0)
                                lon_pt = st.number_input("Longitude (°E)", float(ds[lon_var].min()), float(ds[lon_var].max()), value=60.0)
                                da_sel = da.sel({lat_var: lat_pt, lon_var: lon_pt}, method="nearest")
                                da_sel = da_sel.sel({time_var: slice(t1, t2)})
                                hov_x = da_sel[depth_var]
                                hov_y = da_sel[time_var]
                                hov_z = da_sel.transpose(time_var, depth_var)
                    
                            elif hov_mode == "Depth vs Time • Grid Avg (Lat-Lon box)":
                                col1, col2 = st.columns(2)
                                with col1:
                                    lat_min = st.number_input("Min Latitude", float(ds[lat_var].min()), float(ds[lat_var].max()), value=10.0)
                                    lat_max = st.number_input("Max Latitude", float(ds[lat_var].min()), float(ds[lat_var].max()), value=20.0)
                                with col2:
                                    lon_min = st.number_input("Min Longitude", float(ds[lon_var].min()), float(ds[lon_var].max()), value=50.0)
                                    lon_max = st.number_input("Max Longitude", float(ds[lon_var].min()), float(ds[lon_var].max()), value=70.0)
                                da_sel = da.sel({lat_var: slice(lat_min, lat_max), lon_var: slice(lon_min, lon_max)})
                                da_sel = da_sel.mean(dim=[lat_var, lon_var], skipna=True)
                                da_sel = da_sel.sel({time_var: slice(t1, t2)})
                                hov_x = da_sel[depth_var]
                                hov_y = da_sel[time_var]
                                hov_z = da_sel.transpose(time_var, depth_var)

                            # --- Build subtitle with spatial info ---
                            if hov_mode.startswith("Longitude"):
                                if "Depth-avg" in hov_mode:
                                    spatial_info = f"Lat: {fixed_lat:.2f}°N | Depth: {d1:.0f}–{d2:.0f} m"
                                else:
                                    spatial_info = f"Lat: {fixed_lat:.2f}°N | Depth: {fixed_depth:.0f} m"
                            
                            elif hov_mode.startswith("Latitude"):
                                if "Depth-avg" in hov_mode:
                                    spatial_info = f"Lon: {fixed_lon:.2f}°E | Depth: {d1:.0f}–{d2:.0f} m"
                                else:
                                    spatial_info = f"Lon: {fixed_lon:.2f}°E | Depth: {fixed_depth:.0f} m"
                            
                            elif hov_mode == "Depth vs Time • Fixed Lat & Lon":
                                spatial_info = f"Lat: {lat_pt:.2f}°N | Lon: {lon_pt:.2f}°E"
                            
                            elif hov_mode == "Depth vs Time • Grid Avg (Lat-Lon box)":
                                spatial_info = f"Lat: {lat_min:.0f}–{lat_max:.0f}°N | Lon: {lon_min:.0f}–{lon_max:.0f}°E"
                            
                            else:
                                spatial_info = ""
                            
                            time_range_str = f"{str(t1)[:10]} to {str(t2)[:10]}"
                            
                            # === Plotting ===
                            fig, ax = plt.subplots(figsize=(10, 5))
                            c = ax.contourf(hov_x, hov_y, hov_z, levels=50, cmap=cmap_choice)
                            ax.set_title(f"{var} Hovmöller Diagram\n{hov_mode}\n🗺️ {spatial_info}\n🕒 {time_range_str}", fontsize=13)
                            ax.set_xlabel(hov_x.name)
                            ax.set_ylabel("Time")
                            plt.xticks(rotation=45)
                            fig.colorbar(c, ax=ax, label=var, shrink=0.65)
                            st.pyplot(fig)

                            # time_range_str = f"{str(t1)[:10]} to {str(t2)[:10]}"

                            # # === Plotting ===
                            # fig, ax = plt.subplots(figsize=(10, 5))
                            # c = ax.contourf(hov_x, hov_y, hov_z, levels=50, cmap=cmap_choice)
                            # # ax.set_title(f"{var} Hovmöller Diagram ({hov_mode})", fontsize=14)
                            # ax.set_title(f"{var} Hovmöller Diagram ({hov_mode})\n {time_range_str}", fontsize=14)
                            # ax.set_xlabel(hov_x.name)
                            # ax.set_ylabel("Time")
                            # plt.xticks(rotation=45)
                            # fig.colorbar(c, ax=ax, label=var, shrink=0.65)
                            # st.pyplot(fig)
                    
                            if save_btn:
                                buf = io.BytesIO()
                                fig.savefig(buf, format=save_format, dpi=dpi_value, bbox_inches="tight")
                                st.download_button(
                                    label=f"📥 Download {save_format.upper()}",
                                    data=buf.getvalue(),
                                    file_name=f"{var}_hovmoller_{hov_mode.replace(' ', '_')}.{save_format}",
                                    mime=f"image/{'jpeg' if save_format == 'jpg' else save_format}"
                                )
                    
                        except Exception as e:
                            st.error(f"❌ Failed to plot Hovmöller diagram: {e}")


        except Exception as e:
            st.error(f"⚠️ Failed to subset or plot data: {e}")


        


