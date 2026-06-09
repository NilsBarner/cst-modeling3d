__all__ = ["plot_outer_parallel_edf_axial_ducted_radiator"]

import pyvista as pv
import matplotlib.pyplot as plt


def plot_outer_parallel_edf_axial_ducted_radiator(out_dict):
    
    parallel_edf_mesh = out_dict["parallel_edf"]
    right_axial_ducted_radiator_mesh = out_dict["right_axial_ducted_radiator"]
    left_axial_ducted_radiator_mesh = out_dict["left_axial_ducted_radiator"]
    right_hx_poly = out_dict["right_hx"]
    left_hx_poly = out_dict["left_hx"]
    right_fan_poly = out_dict["right_fan"]
    left_fan_poly = out_dict["left_fan"]
    
    #%% Display meshes in one window

    # Clip master mesh to negative y-values and extract slice
    master_mesh_clip = parallel_edf_mesh.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )
    master_mesh_slice = parallel_edf_mesh.slice(
        normal=(0,1,0),
        origin=(0,0,0),
    )

    # plotter = pv.Plotter()
    # plotter.add_mesh(parallel_edf_mesh, show_edges=False, color='blue', opacity=0.5)
    # # plotter.add_mesh(left_axial_ducted_radiator_mesh, show_edges=False, color='orange', opacity=0.5)
    # plotter.add_mesh(right_axial_ducted_radiator_mesh, show_edges=False, color='orange', opacity=0.5)
    # plotter.add_axes()
    # plotter.show_grid()
    # plotter.show()

    #%%

    # Plot all features

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(1600, 1200),
        image_scale=3,
    )

    plotter.add_mesh(parallel_edf_mesh, show_edges=False, color='lightgrey', opacity=0.1)
    # plotter.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.1)
    # plotter.add_mesh(master_mesh_slice, color="black", line_width=2)

    # plotter.add_mesh(outer_poly_pos_edf, color="grey", opacity=0.95)
    # plotter.add_mesh(outer_poly_neg_edf, color="grey", opacity=0.95)

    # plotter.add_mesh(core_poly_edf, color="grey", opacity=0.95)

    plotter.add_mesh(right_axial_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.4)
    plotter.add_mesh(left_axial_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.4)

    # plotter.add_mesh(hx_poly_pos, color="orange", opacity=0.4)
    # plotter.add_mesh(hx_poly_neg, color="orange", opacity=0.4)
    plotter.add_mesh(right_hx_poly, show_edges=False, color='orange', opacity=0.95)
    plotter.add_mesh(left_hx_poly, show_edges=False, color='orange', opacity=0.95)

    plotter.add_mesh(right_fan_poly, show_edges=False, color='red', opacity=0.95)
    plotter.add_mesh(left_fan_poly, show_edges=False, color='red', opacity=0.95)

    # views = [
    #     (0, 0),
    #     (90, 0),
    #     (180, 0),
    #     (-90, 0),
    #     (0, 180),
    #     (90, 180),
    #     (180, 180),
    #     (-90, 180),
    # ]

    # for i, (az, el) in enumerate(views):

    plotter.view_isometric()
    # plotter.camera.Azimuth(az)
    # plotter.camera.Elevation(el)
    # plotter.camera.Azimuth(0)
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
    
    ### NILS: original content of outer_parallel_edf_axial_ducted_radiator_plotter.py to plot from disk

    # dat_files = glob.glob(os.path.join(
    #     r'C:\Users\nmb48\Documents\GitHub\Flydrogen\flydrogen\systems\edf\cst_plotting\parallel',
    #     'parallel_edf_mesh_[0-9]*.dat',
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



