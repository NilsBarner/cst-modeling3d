__all__ = ["plot_bypass_edf"]

import pyvista as pv
import matplotlib.pyplot as plt


def plot_bypass_edf(out_dict):
    
    bypass_edf_mesh = out_dict["master"]
    outer_poly_pos = out_dict["outer_pos"]
    outer_poly_neg = out_dict["outer_neg"]
    inner_poly_pos = out_dict["inner_pos"]
    inner_poly_neg = out_dict["inner_neg"]
    hx_poly_pos = out_dict["hx_pos"]
    hx_poly_neg = out_dict["hx_neg"]
    hx_poly = out_dict["hx"]
    core_poly = out_dict["core_poly"]
    
    # plotter = pv.Plotter()
    # plotter.add_mesh(bypass_edf_mesh, show_edges=True, color='lightgrey', opacity=1.0)
    # plotter.add_axes()
    # plotter.show_grid()
    # plotter.show()
    
    #%%
    
    # Clip master mesh to negative y-values and extract slice
    master_mesh_clip = bypass_edf_mesh.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )
    master_mesh_slice = bypass_edf_mesh.slice(
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
    
    plotter.add_mesh(inner_poly_pos, color="grey", opacity=0.95)
    plotter.add_mesh(inner_poly_neg, color="grey", opacity=0.95)
    
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
    
    ### NILS: original content of bypasss_edf_plotter.py to plot from disk

    # dat_files = glob.glob(os.path.join(
    #     r'C:\Users\nmb48\Documents\GitHub\Flydrogen\flydrogen\systems\edf\cst_plotting\bypass',
    #     'surface_[0-9]*.dat',
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

