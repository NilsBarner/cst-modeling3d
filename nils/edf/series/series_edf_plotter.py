__all__ = ["plot_series_edf"]

import pyvista as pv
import matplotlib.pyplot as plt


def plot_series_edf(out_dict):
    
    series_edf_mesh = out_dict["master"]
    outer_poly_pos = out_dict["outer_pos"]
    outer_poly_neg = out_dict["outer_neg"]
    core_poly = out_dict["core_poly"]
    hx_poly_pos = out_dict["hx_pos"]
    hx_poly_neg = out_dict["hx_neg"]
    hx_poly = out_dict["hx"]
    core_mesh = out_dict["core"]
    
    # Clip master mesh to negative y-values and extract slice
    master_mesh_clip = series_edf_mesh.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )
    master_mesh_slice = series_edf_mesh.slice(
        normal=(0,1,0),
        origin=(0,0,0),
    )
    hx_poly_clip = hx_poly.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )

    # Plot all features

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(1600, 1200),
        image_scale=3,
    )

    plotter.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.2)
    plotter.add_mesh(master_mesh_slice, color="black", line_width=2)

    plotter.add_mesh(outer_poly_pos, color="grey", opacity=0.95)
    plotter.add_mesh(outer_poly_neg, color="grey", opacity=0.95)

    plotter.add_mesh(core_poly, color="grey", opacity=0.95)

    # plotter.add_mesh(hx_poly_pos, color="orange", opacity=0.4)
    # plotter.add_mesh(hx_poly_neg, color="orange", opacity=0.4)
    plotter.add_mesh(hx_poly_clip, show_edges=False, color='orange', opacity=0.4)

    # views = [
    #     # (0, 0),
    #     # (90, 0),
    #     # (180, 0),
    #     (-90, 0),
    #     # (0, 180),
    #     # (90, 180),
    #     # (180, 180),
    #     # (-90, 180),
    # ]

    # for i, (az, el) in enumerate(views):

    plotter.view_isometric()
    # plotter.camera.Azimuth(az)
    # plotter.camera.Elevation(el)
    plotter.camera.Azimuth(-90)
    plotter.camera.Elevation(0)

    plotter.reset_camera()  # <-- prevents the zoom issue

    plotter.show(auto_close=False)
    plotter.screenshot(f"img.png", scale=3)

    img = plotter.image
    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(plotter.image)
    plt.show()
        
    #%%

    plotter = pv.Plotter()
    plotter.add_mesh(core_mesh)
    plotter.show()

    #%%

    # =============================================================================
    from nils.bin_packing.shared_functions import convert_core_mesh_to_watertight_surface_mesh
    watertight_core_mesh = convert_core_mesh_to_watertight_surface_mesh(core_mesh)
    # =============================================================================

    from nils.bin_packing.shared_functions import pv_to_trimesh
    tm = pv_to_trimesh(watertight_core_mesh)

    plotter = pv.Plotter()
    plotter.add_mesh(watertight_core_mesh)
    plotter.show()

    print('tm.is_watertight =', tm.is_watertight)
    
    ### NILS: original content of series_edf_plotter.py to plot from disk

    # dat_files = glob.glob(os.path.join(
    #     r'C:\Users\nmb48\Documents\GitHub\Flydrogen\flydrogen\systems\edf\cst_plotting\series',
    #     'series_edf_mesh_[0-9]*.dat',
    # ))
    
    # # Create a plotter
    # plotter = pv.Plotter()
    
    # for dat_file in dat_files:
    #     mesh = pv.read(dat_file)
    #     plotter.add_mesh(mesh, color='lightblue')
    
    # # Optional border
    # plotter.show_bounds(color='k')
    
    # # Show the plot
    # plotter.show()
    
    return


