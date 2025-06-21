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
    
st.set_page_config(layout="wide")
st.title("🌊 Ocean Viewer")


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

def load_netcdf_safe_from_path(path):
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except ValueError as e:
        if "unable to decode time units" in str(e) and "calendar 'NOLEAP'" in str(e):
            st.warning("⚠️ Time decoding failed. Retrying with decode_times=False...")
            return xr.open_dataset(path, decode_times=False, engine="netcdf4")
        else:
            raise

# @st.cache_data
# def load_netcdf_safe_from_path(path):
#     import os
#     import xarray as xr

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"File does not exist: {path}")
#     if not path.endswith(".nc"):
#         raise ValueError("Not a NetCDF file")

#     try:
#         return xr.open_dataset(path, engine="netcdf4")
#     except ValueError as e:
#         if "unable to decode time units" in str(e) and "calendar 'NOLEAP'" in str(e):
#             st.warning("⚠️ Time decoding failed. Retrying with decode_times=False...")
#             return xr.open_dataset(path, decode_times=False, engine="netcdf4")
#         else:
#             raise


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

# --- File uploader ---
uploaded_file = st.file_uploader("📂 Upload a NetCDF file", type=["nc"])

if uploaded_file:
    ds = load_netcdf_safe(uploaded_file)

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

        var = st.selectbox("🔎 Variable", list(plot_vars.keys()))
        ds_sel = ds[var]

        # Ensure dims are coords
        for d in ds_sel.dims:
            if d not in ds_sel.coords and d in ds.coords:
                ds_sel = ds_sel.assign_coords({d: ds[d]})
                
        #------------------------LEFT SIDE - INPUTS-------------------------------------#
        left_col, right_col = st.columns([1, 2])

        with left_col:
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

            lat_range = st.slider("🌐 Latitude Range", float(lat_vals.min()), float(lat_vals.max()), (float(lat_vals.min()), float(lat_vals.max())))
            lon_range = st.slider("🗺️ Longitude Range", float(lon_vals.min()), float(lon_vals.max()), (float(lon_vals.min()), float(lon_vals.max())))

            if depth_var:
                depth_vals = ds[depth_var].values
                selected_depth = st.slider("🧭 Select Depth Level", float(depth_vals.min()), float(depth_vals.max()), float(depth_vals.min())) if len(depth_vals) > 1 else depth_vals[0]
            else:
                selected_depth = None

            if time_var:
                raw_time_vals, time_labels = try_decode_time(ds, time_var)
                time_sel = st.selectbox("🕒 Select Time", time_labels)
                try:
                    time_index = list(time_labels).index(time_sel)
                    raw_time_value = raw_time_vals[time_index]
                except Exception:
                    raw_time_value = raw_time_vals[0]
            else:
                raw_time_value = None
                
            # Use left_col for plot selection checkboxes
            with st.expander("🗺️ Select Plot Options", expanded=True):
                show_spatial_map = st.checkbox("Spatial Map")
                show_interactive_spatial_map = st.checkbox("Spatial Interactive Map")
                show_time_animation = st.checkbox("Spatial Map - Time Animation")
                show_vertical_profile = st.checkbox("Vertical Profile (Single Location)")
                show_interactive_vertical_profile =  st.checkbox("Vertical Interactive Profile (Single Location)")

            if show_vertical_profile or show_interactive_vertical_profile:
                st.markdown("### 📍 Vertical Profile Location")
                input_lat = st.number_input("Latitude", value=0.0, format="%.2f")
                input_lon = st.number_input("Longitude", value=60.0, format="%.2f")
                # if st.button("Extract Vertical Profile"):
                #     trigger_profile_plot = True
                # else:
                #     trigger_profile_plot = False
                    
            if show_spatial_map or show_time_animation or show_interactive_spatial_map:
                with st.expander("🌍 Land/Sea Masking"):
                    mask_land = st.checkbox("Mask Land", value=False)
                    mask_sea = st.checkbox("Mask Ocean", value=False)
                    mask_color = st.selectbox("Mask Color", ["lightgray", "gray", "black", "white", "skyblue", "khaki", "coral", "forestgreen"])
            
            def reset_colorbar_settings():
                st.session_state["set_clim"] = False
                for key in ["vmin", "vmax", "step", "cmap_choice"]:
                    st.session_state.pop(key, None)
                st.session_state["cmap_choice"] = "viridis"

            if show_spatial_map or show_time_animation or show_interactive_spatial_map:
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

            def reset_plot_labels(time_sel_value=None, depth_value=None):
                title = var
                if time_sel_value:
                    title += f" | {time_sel_value}"
                if depth_value is not None:
                    title += f" | {depth_value} m"
                st.session_state["plot_title"] = title
                st.session_state["xlabel"] = "Longitude"
                st.session_state["ylabel"] = "Latitude"
                st.session_state["cbar_label"] = var

            with st.expander("🖊️ Plot Custom Labels"):
                label_cols = st.columns([2, 1])
                with label_cols[0]:
                    plot_title = st.text_input("📌 Plot Title", value=var, key="plot_title")
                    xlabel = st.text_input("🧭 X-axis Label", value="Longitude", key="xlabel")
                    ylabel = st.text_input("🧭 Y-axis Label", value="Latitude", key="ylabel")
                    cbar_label = st.text_input("🎨 Colorbar Label", value=var, key="cbar_label")
                with label_cols[1]:
                    # st.button("🔄 Reset", on_click=reset_plot_labels)
                    # st.button("🔄 Reset", on_click=reset_plot_labels, key="reset_plot_labels_btn")
                    st.button("🔄 Reset", on_click=lambda: reset_plot_labels(time_sel, selected_depth), key="reset_plot_labels_btn")


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
                
        #------------------------RIGHT SIDE - OUTPUTS-------------------------------------#
        with right_col:
            st.subheader("📄 Dataset Structure")
            st.code(ds.__repr__(), language="python")

            # --- Subset and Plot ---
            subset_kwargs = {}
            if time_var and time_var in ds_sel.dims and raw_time_value is not None:
                ds_sel = ds_sel.sel({time_var: raw_time_value}, method="nearest")
            if depth_var and depth_var in ds_sel.dims and selected_depth is not None:
                ds_sel = ds_sel.sel({depth_var: selected_depth}, method="nearest")
            if lat_var in ds_sel.dims:
                subset_kwargs[lat_var] = slice(*lat_range)
            if lon_var in ds_sel.dims:
                subset_kwargs[lon_var] = slice(*lon_range)
            data = ds_sel.sel(subset_kwargs)

            #---------------------------------Normal Map View----------------------------------------------------------#
            
            if show_spatial_map:
                st.subheader("🗺️ Map View")
                # Apply the selected font family first
                plt.rcParams['font.family'] = st.session_state.get("font_family", "DejaVu Sans")                
    
                fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                plot_kwargs = {"ax": ax, "transform": ccrs.PlateCarree(), "cmap": cmap_choice, "add_colorbar": True}
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
    
                ax.set_title(plot_title, fontsize=14)
                if hasattr(im, 'colorbar') and im.colorbar:
                    im.colorbar.set_label(cbar_label, fontsize=12)
                ax.text(0.5, -0.1, xlabel, transform=ax.transAxes, ha='center', va='top', fontsize=12)
                ax.text(-0.15, 0.5, ylabel, transform=ax.transAxes, ha='right', va='center', rotation='vertical', fontsize=12)
                st.pyplot(fig)
    
    
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

            #---------------------------------Intercative Map View----------------------------------------------------------#
            
            if show_interactive_spatial_map:
                st.subheader("🎞️ Interactive Map View")
    
                def figsize_to_plotly(width_in, height_in, dpi=100):
                    return int(width_in * dpi), int(height_in * dpi)
    
                import streamlit as st
                import xarray as xr
                import plotly.graph_objects as go
    
                def standardize_coords(dataarray):
                    coord_map = {
                        'latitude': None,
                        'longitude': None,
                        'time': None,
                        'depth': None
                    }
                
                    coord_candidates = {k.lower(): k for k in dataarray.coords}
                
                    # Match based on known naming variants
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
                
    
                # Assuming `data` is your 2D DataArray (lat x lon)
                data_2d = data.squeeze()
                # st.write("Data coordinates:", data_2d.coords)
                
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
                    title=plot_title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    # height=600,
                    width=width,
                    height=height
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            #-------------------------------------------------------------------------------------------------------------------#
            
            if show_time_animation:
                # === Create Animated Plot over Time ===
                import matplotlib.animation as animation
                import io
                
                st.subheader("🎞️ Time-Loop Animation (GIF)")
                plt.rcParams['font.family'] = st.session_state.get("font_family", "DejaVu Sans")
                
                if time_var and time_var in ds[var].dims:
                    try:
                        da_anim = ds[var]
                
                        if depth_var and selected_depth is not None and depth_var in da_anim.dims:
                            da_anim = da_anim.sel({depth_var: selected_depth}, method="nearest")
                
                        da_anim = da_anim.sel({lat_var: slice(*lat_range), lon_var: slice(*lon_range)})
                
                        fig_anim, ax_anim = plt.subplots(figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()})
                        
                        # --- Draw colorbar once outside animation loop ---
                        first_frame = da_anim.isel({time_var: 0})
                        im_cbar = first_frame.plot.pcolormesh(
                            ax=ax_anim,
                            transform=ccrs.PlateCarree(),
                            cmap=cmap_choice,
                            vmin=vmin if set_clim else None,
                            vmax=vmax if set_clim else None,
                            add_colorbar=False
                        )
                        cbar = fig_anim.colorbar(im_cbar, ax=ax_anim, orientation="vertical", shrink=0.7, pad=0.05, extend='both')
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
                            gl.top_labels = False
                            gl.right_labels = False
                            gl.xlabel_style = {'size': 10}
                            gl.ylabel_style = {'size': 10}
                        
                            if st.session_state.get("manual_ticks", False):
                                xtick_step = st.session_state.get("xtick_step")
                                ytick_step = st.session_state.get("ytick_step")
                                if xtick_step and ytick_step:
                                    gl.xlocator = mticker.FixedLocator(np.arange(lon_range[0], lon_range[1] + xtick_step, xtick_step))
                                    gl.ylocator = mticker.FixedLocator(np.arange(lat_range[0], lat_range[1] + ytick_step, ytick_step))
                        
                            ax_anim.text(0.5, -0.1, xlabel, transform=ax_anim.transAxes, ha='center', va='top', fontsize=10)
                            ax_anim.text(-0.15, 0.5, ylabel, transform=ax_anim.transAxes, ha='right', va='center', rotation='vertical', fontsize=10)
                        
                            # 🕒 Use decoded, formatted time string from time_labels
                            try:
                                time_str = pd.to_datetime(time_labels[frame]).strftime("%Y-%m-%d")
                            except:
                                time_str = str(time_labels[frame])[:15]
                        
                            title = f"{plot_title}"# | Time: {time_str}"
                            # if depth_var and selected_depth is not None:
                            #     title += f" | Depth: {selected_depth} m"
                        
                            ax_anim.set_title(title, fontsize=12)
                            return [im]
    
                        # fig.tight_layout()
                        # fig_anim.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
                        fig_anim.subplots_adjust(left=0.01, right=1)
    
    
                        ani = animation.FuncAnimation(
                            fig_anim, update_anim, frames=da_anim.sizes[time_var], blit=False
                        )
                        
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                            temp_gif_path = tmpfile.name
                        
                        ani.save(temp_gif_path, writer="pillow", fps=2, savefig_kwargs={'bbox_inches': 'tight'})
                        
                        # Display the animation in Streamlit
                        with open(temp_gif_path, "rb") as f:
                            gif_bytes = f.read()
                        
                        st.image(gif_bytes, caption="Time-animated plot", use_container_width=True)
                        
                        st.download_button(
                            label="📥 Download GIF",
                            data=gif_bytes,  # ✅ Use directly, no .getvalue()
                            file_name=f"{var}_animation.gif",
                            mime="image/gif"
                        )
                        
                        # Optional cleanup
                        os.remove(temp_gif_path)
                
                    except Exception as e:
                        st.error(f"⚠️ Failed to create animation: {e}")
                else:
                    st.info("⏳ Animation unavailable: Time dimension not found in selected variable.")

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

            # -----------------------Interactive Vetical Profile --------------------------------#
            import plotly.graph_objects as go
            if show_interactive_vertical_profile:
                st.markdown("### 📉 Vertical Profile")
            
                # Reuse coord map for robustness
                coord_map = detect_coord_names(ds_sel)
            
                lat_key = coord_map["latitude"]
                lon_key = coord_map["longitude"]
                depth_key = coord_map["depth"]
                time_key = coord_map["time"]
            
                if not all([lat_key, lon_key, depth_key]):
                    st.error("❌ Could not detect necessary coordinate names (lat/lon/depth).")
                else:
                    # Build selection dictionary
                    sel_dict = {
                        lat_key: input_lat,
                        lon_key: input_lon
                    }
            
                    if time_key and time_sel is not None:
                        sel_dict[time_key] = time_sel
            
                    try:
                        profile = ds[var].sel(sel_dict, method="nearest")
            
                        # Handle depth as coordinate or dimension
                        if depth_key in profile.coords:
                            depth_vals = profile[depth_key].values
                        elif depth_key in ds.coords:
                            depth_vals = ds[depth_key].values
                        else:
                            st.error("❌ Could not find depth values in dataset.")
                            st.stop()
            
                        var_vals = profile.values
            
                        # Safety check for matching shapes
                        if depth_vals.shape[0] != var_vals.shape[0]:
                            st.warning("⚠️ Profile and depth shapes may not match. Check dimensions.")
            
                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=depth_vals,
                            x=var_vals,
                            mode='lines+markers',
                            name=var
                        ))
                        fig.update_layout(
                            title=f"{var} Profile at ({input_lat:.2f}, {input_lon:.2f})",
                            xaxis_title=var,
                            yaxis_title="Depth (m)",
                            yaxis_autorange="reversed",
                            height=500,
                            width=500
                        )
                        st.plotly_chart(fig)
            
                    except Exception as e:
                        st.error(f"❌ Failed to extract profile: {e}")

            
            #---------------------------Normal Vertical Profile ----------------------------------#
            if show_vertical_profile:
                st.markdown("### 📉 Vertical Profile")
            
                # Reuse coordinate map for flexibility
                coord_map = detect_coord_names(ds_sel)
            
                lat_key = coord_map["latitude"]
                lon_key = coord_map["longitude"]
                depth_key = coord_map["depth"]
                time_key = coord_map["time"]
            
                if not all([lat_key, lon_key, depth_key]):
                    st.error("❌ Could not detect necessary coordinate names (lat/lon/depth).")
                else:
                    sel_dict = {
                        lat_key: input_lat,
                        lon_key: input_lon
                    }
            
                    if time_key and time_sel is not None:
                        sel_dict[time_key] = time_sel
            
                    try:
                        profile = ds[var].sel(sel_dict, method="nearest")
            
                        # Get depth values
                        if depth_key in profile.coords:
                            depth_vals = profile[depth_key].values
                        elif depth_key in ds.coords:
                            depth_vals = ds[depth_key].values
                        else:
                            st.error("❌ Could not find depth values in dataset.")
                            st.stop()
            
                        var_vals = profile.values
            
                        if depth_vals.shape[0] != var_vals.shape[0]:
                            st.warning("⚠️ Depth and variable value shapes may not match.")
            
                        # Matplotlib plot
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.plot(var_vals, depth_vals, marker='o', linestyle='-')
                        ax.set_xlabel(var)
                        ax.set_ylabel("Depth (m)")
                        ax.invert_yaxis()
                        ax.set_title(f"{var} Profile at ({input_lat:.2f}, {input_lon:.2f})")
            
                        st.pyplot(fig)
            
                    except Exception as e:
                        st.error(f"❌ Failed to extract profile: {e}")



                
        # except Exception as e:
        #     st.error(f"⚠️ Failed to subset or plot data: {e}")


        


