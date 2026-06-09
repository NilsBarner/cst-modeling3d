__all__ = ["plot_inner_parallel_edf_axial_ducted_radiator"]

import pyvista as pv
import matplotlib.pyplot as plt


def plot_inner_parallel_edf_axial_ducted_radiator(out_dict):
    
    parallel_edf_mesh = out_dict["parallel_edf"]
    outer_poly_pos_edf = out_dict["outer_pos_edf"]
    outer_poly_neg_edf = out_dict["outer_neg_edf"]
    core_poly_edf = out_dict["core_edf"]
    axial_ducted_radiator_mesh = out_dict["axial_ducted_radiator"]
    hx_poly = out_dict["hx"]
    hx_poly_pos = out_dict["hx_pos"]
    hx_poly_neg = out_dict["hx_neg"]
    fan_poly = out_dict["fan"]
    fan_poly_pos = out_dict["fan_pos"]
    fan_poly_neg = out_dict["fan_neg"]
    outer_poly_pos = out_dict["outer_pos"]
    outer_poly_neg = out_dict["outer_neg"]
    
    #%% Display meshes in one window

    master_mesh = pv.merge([parallel_edf_mesh, axial_ducted_radiator_mesh])

    # Clip master mesh to negative y-values and extract slice
    master_mesh_clip = master_mesh.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )
    master_mesh_slice = master_mesh.slice(
        normal=(0,1,0),
        origin=(0,0,0),
    )
    hx_poly_clip = hx_poly.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )
    fan_poly_clip = fan_poly.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )

    plotter = pv.Plotter()

    plotter.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.4)
    plotter.add_mesh(master_mesh_slice, color="black", line_width=3)

    plotter.add_mesh(hx_poly_pos, show_edges=False, color='orange', opacity=0.4)
    plotter.add_mesh(hx_poly_neg, show_edges=False, color='orange', opacity=0.4)

    plotter.add_mesh(fan_poly_pos, show_edges=False, color='red', opacity=0.4)
    plotter.add_mesh(fan_poly_neg, show_edges=False, color='red', opacity=0.4)

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

    #%%

    # Plot all features

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(1600, 1200),
        image_scale=3,
    )

    plotter.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.2)
    plotter.add_mesh(master_mesh_slice, color="black", line_width=2)

    plotter.add_mesh(outer_poly_pos_edf, color="grey", opacity=0.95)
    plotter.add_mesh(outer_poly_neg_edf, color="grey", opacity=0.95)

    plotter.add_mesh(outer_poly_pos, color="grey", opacity=0.95)
    plotter.add_mesh(outer_poly_neg, color="grey", opacity=0.95)

    plotter.add_mesh(core_poly_edf, color="grey", opacity=0.95)

    # plotter.add_mesh(hx_poly_pos, color="orange", opacity=0.4)
    # plotter.add_mesh(hx_poly_neg, color="orange", opacity=0.4)
    plotter.add_mesh(hx_poly_clip, show_edges=False, color='orange', opacity=0.4)

    # plotter.add_mesh(fan_poly_pos, color="red", opacity=0.4)
    # plotter.add_mesh(fan_poly_neg, color="red", opacity=0.4)
    plotter.add_mesh(fan_poly_clip, show_edges=False, color='red', opacity=0.4)

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
    
    return



    