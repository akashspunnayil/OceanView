import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tempfile
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker

    
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


            with st.expander("🌍 Land/Sea Masking"):
                mask_land = st.checkbox("Mask Land", value=False)
                mask_sea = st.checkbox("Mask Ocean", value=False)
            
                mask_color = st.selectbox(
                    "Mask Color",
                    options=["lightgray", "gray", "black", "white", "skyblue", "khaki", "coral", "forestgreen"],
                    index=0
                )

            def reset_colorbar_settings():
                st.session_state["set_clim"] = False
                st.session_state.pop("vmin", None)
                st.session_state.pop("vmax", None)
                st.session_state.pop("step", None)
                st.session_state["cmap_choice"] = "viridis"
            
            with st.expander("🎨 Colorbar & Colormap Settings"):
                cols_colorbar = st.columns([2, 1])  # settings + reset button
            
                with cols_colorbar[0]:
                    set_clim = st.checkbox("🔧 Manually set colorbar range", key="set_clim")
            
                    if set_clim:
                        vmin = st.number_input("Minimum value (vmin)", value=0.0, key="vmin")
                        vmax = st.number_input("Maximum value (vmax)", value=1.0, key="vmax")
                        step = st.number_input("Tick interval (optional)", value=0.1, key="step")
                    else:
                        vmin, vmax, step = None, None, None
            
                    cmap_choice = st.selectbox(
                        "🎨 Choose a colormap",
                        options=sorted([
                            "viridis", "plasma", "inferno", "magma", "cividis",
                            "jet", "turbo", "coolwarm", "RdBu_r", "YlGnBu", "BrBG", "bwr"
                        ]),
                        index=0,
                        key="cmap_choice"
                    )
            
                with cols_colorbar[1]:
                    # st.button("🔄 Reset", on_click=reset_colorbar_settings)
                    st.button("🔄 Reset", on_click=reset_colorbar_settings, key="reset_colorbar_btn")
                    

            def reset_plot_labels():
                # Generate dynamic title based on selections
                title = var
                if time_var:
                    title += f" | {time_sel}"
                if depth_var and selected_depth is not None:
                    title += f" | {selected_depth} m"
            
                # Set default labels
                st.session_state["plot_title"] = title
                st.session_state["xlabel"] = "Longitude"
                st.session_state["ylabel"] = "Latitude"
                st.session_state["cbar_label"] = var


            with st.expander("🖊️ Plot Custom Labels"):
                label_cols = st.columns([2, 1])  # Text inputs and reset
            
                with label_cols[0]:
                    plot_title = st.text_input("📌 Plot Title", value="Ocean Variable Plot", key="plot_title")
                    xlabel = st.text_input("🧭 X-axis Label", value="Longitude", key="xlabel")
                    ylabel = st.text_input("🧭 Y-axis Label", value="Latitude", key="ylabel")
                    cbar_label = st.text_input("🎨 Colorbar Label", value=var, key="cbar_label")
            
                with label_cols[1]:
                    st.button("🔄 Reset", on_click=reset_plot_labels, key="reset_labels_btn")


            # --- Define reset function ---
            def reset_tick_settings():
                if "manual_ticks" in st.session_state:
                    st.session_state["manual_ticks"] = False
                st.session_state.pop("xtick_step", None)
                st.session_state.pop("ytick_step", None)
            
            # --- Axis Tick Settings UI ---
            with st.expander("📏 Axis Tick Settings"):
                tick_cols = st.columns([2, 1])  # input fields | reset button
            
                with tick_cols[0]:
                    manual_ticks = st.checkbox("🔧 Manually set tick intervals", key="manual_ticks")
            
                    xtick_step = None
                    ytick_step = None
            
                    if manual_ticks:
                        xtick_step = st.number_input(
                            "Longitude Tick Interval (°)", 
                            min_value=0.1, max_value=60.0, value=10.0, step=1.0, key="xtick_step"
                        )
                        ytick_step = st.number_input(
                            "Latitude Tick Interval (°)", 
                            min_value=0.1, max_value=60.0, value=5.0, step=1.0, key="ytick_step"
                        )
            
                with tick_cols[1]:
                    st.button("🔄 Reset", on_click=reset_tick_settings)

                
            with st.expander("💾 Save Plot Options"):
                save_format = st.selectbox("Select file format", ["png", "jpg", "pdf", "svg", "tiff"], index=0)
                dpi_value = st.number_input("DPI (dots per inch)", min_value=50, max_value=600, value=150, step=10)
                save_btn = st.button("💾 Save & Download Plot")

            

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
                
                # Plotting the data
                im = data.squeeze().plot.pcolormesh(**plot_kwargs)
                
                # Add coastlines
                ax.coastlines()
                
                # Add gridlines and enable coordinate labels
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 12}
                gl.ylabel_style = {'size': 12}
                
                if st.session_state.get("manual_ticks", False):
                    xtick_step = st.session_state.get("xtick_step", None)
                    ytick_step = st.session_state.get("ytick_step", None)
                
                    if xtick_step and ytick_step:
                        gl.xlocator = mticker.FixedLocator(np.arange(lon_range[0], lon_range[1] + xtick_step, xtick_step))
                        gl.ylocator = mticker.FixedLocator(np.arange(lat_range[0], lat_range[1] + ytick_step, ytick_step))


                import cartopy.feature as cfeature

                if mask_land:
                    ax.add_feature(cfeature.LAND, facecolor=mask_color, zorder=3)
                if mask_sea:
                    ax.add_feature(cfeature.OCEAN, facecolor=mask_color, zorder=3)
                
                # Title
                ax.set_title(plot_title, fontsize=14)
                
                # Colorbar label
                if hasattr(im, 'colorbar') and im.colorbar:
                    im.colorbar.set_label(cbar_label, fontsize=12)

                # Add custom axis labels manually
                ax.text(0.5, -0.1, xlabel, transform=ax.transAxes, ha='center', va='top', fontsize=12)
                ax.text(-0.15, 0.5, ylabel, transform=ax.transAxes, ha='right', va='center', rotation='vertical', fontsize=12)

                # Optional subtitle
                title = f"{plot_title}"
                if time_var: title += f" | Time: {time_sel}"
                if depth_var: title += f" | Depth: {selected_depth} m"
                
                st.pyplot(fig)


                import io

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
            
                # === Create Animated Plot over Time ===
                import matplotlib.animation as animation
                import io
                
                st.subheader("🎞️ Time-Loop Animation (GIF)")
                
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
                                xtick_step = st.session_state.get("xtick_step", None)
                                ytick_step = st.session_state.get("ytick_step", None)
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
                        
                            title = f"{var} | Time: {time_str}"
                            if depth_var and selected_depth is not None:
                                title += f" | Depth: {selected_depth} m"
                        
                            ax_anim.set_title(title, fontsize=12)
                            return [im]

                        # def update_anim(frame):
                        #     ax_anim.clear()
                        #     frame_data = da_anim.isel({time_var: frame})
                            
                        #     im = frame_data.plot.pcolormesh(
                        #         ax=ax_anim,
                        #         transform=ccrs.PlateCarree(),
                        #         cmap=cmap_choice,
                        #         vmin=vmin if set_clim else None,
                        #         vmax=vmax if set_clim else None,
                        #         add_colorbar=False  # handle colorbar only once outside
                        #     )
                            
                        #     # Coastlines and mask
                        #     ax_anim.coastlines()
                        #     if mask_land:
                        #         ax_anim.add_feature(cfeature.LAND, facecolor=mask_color, zorder=3)
                        #     if mask_sea:
                        #         ax_anim.add_feature(cfeature.OCEAN, facecolor=mask_color, zorder=3)
                            
                        #     # Gridlines
                        #     gl = ax_anim.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                        #     gl.top_labels = False
                        #     gl.right_labels = False
                        #     gl.xlabel_style = {'size': 10}
                        #     gl.ylabel_style = {'size': 10}
                            
                        #     # Apply manual tick intervals if set
                        #     if st.session_state.get("manual_ticks", False):
                        #         xtick_step = st.session_state.get("xtick_step", None)
                        #         ytick_step = st.session_state.get("ytick_step", None)
                        #         if xtick_step and ytick_step:
                        #             gl.xlocator = mticker.FixedLocator(np.arange(lon_range[0], lon_range[1] + xtick_step, xtick_step))
                        #             gl.ylocator = mticker.FixedLocator(np.arange(lat_range[0], lat_range[1] + ytick_step, ytick_step))
                            
                        #     # Axis labels (as text because Cartopy disables set_xlabel)
                        #     ax_anim.text(0.5, -0.1, xlabel, transform=ax_anim.transAxes, ha='center', va='top', fontsize=10)
                        #     ax_anim.text(-0.15, 0.5, ylabel, transform=ax_anim.transAxes, ha='right', va='center', rotation='vertical', fontsize=10)
                        
                        #     # Title with time + depth
                        #     time_value = da_anim[time_var].isel({time_var: frame}).values
                        #     try:
                        #         time_str = pd.to_datetime(str(time_value)).strftime("%Y-%m-%d")
                        #     except:
                        #         time_str = str(time_value)[:15]
                            
                        #     title = f"{var} | Time: {time_str}"
                        #     if depth_var and selected_depth is not None:
                        #         title += f" | Depth: {selected_depth} m"
                            
                        #     ax_anim.set_title(title, fontsize=12)
                        #     return [im]


                        ani = animation.FuncAnimation(
                            fig_anim, update_anim, frames=da_anim.sizes[time_var], blit=False
                        )
                        
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                            temp_gif_path = tmpfile.name
                        
                        ani.save(temp_gif_path, writer="pillow", fps=2)
                        
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

                    
            except Exception as e:
                st.error(f"⚠️ Failed to subset or plot data: {e}")


        


