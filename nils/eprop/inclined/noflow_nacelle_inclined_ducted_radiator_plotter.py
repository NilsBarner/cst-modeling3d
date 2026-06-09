__all__ = ["plot_noflow_nacelle_inclined_ducted_radiator"]

import pyvista as pv


def plot_noflow_nacelle_inclined_ducted_radiator(out_dict):
    
    inclined_ducted_radiator_mesh = out_dict["inclined_ducted_radiator"]
    noflow_nacelle_mesh = out_dict["noflow_nacelle"]
    spinner_mesh = out_dict["spinner"]
    actuator_disk_mesh = out_dict["actuator_disk"]
    
    #%% Clip duct mesh with nacelle mesh
    # NOTE: boolean surface operations in pyvista are only defined for
    # PolyData, not UnstructuredGrid (result of pv.merge())
    # 1) Convert both meshes to surface PolyData
    # 2) Triangulate them
    # 3) Perform the boolean operation
    # NOTE: the boolean_intersection() call takes some time to complete

    # large = noflow_nacelle_mesh.extract_surface().triangulate()
    # small = inclined_ducted_radiator_mesh.extract_surface().triangulate()
    # clipped_inclined_ducted_radiator_mesh = small.boolean_intersection(large)

    #%% Display meshes in one window

    mask_fan = inclined_ducted_radiator_mesh.cell_data['region'] == 3
    submesh_fan = inclined_ducted_radiator_mesh.extract_cells(mask_fan)

    mask_hx = inclined_ducted_radiator_mesh.cell_data['region'] == 1
    submesh_hx = inclined_ducted_radiator_mesh.extract_cells(mask_hx)

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(1600, 1200),
        image_scale=3,
    )
    plotter.add_mesh(noflow_nacelle_mesh, show_edges=False, color='lightgrey', opacity=0.2)
    plotter.add_mesh(spinner_mesh, show_edges=False, color='lightgrey', opacity=0.2)
    plotter.add_mesh(actuator_disk_mesh, show_edges=False, color='lightgrey', opacity=0.2)
    # plotter.add_mesh(inclined_ducted_radiator_mesh, show_edges=False, color='orange', opacity=0.2)
    # =============================================================================
    # # =============================================================================
    # # Define a categorical colormap
    # region_colors = ['red', 'green', 'blue', 'orange', 'purple']  # one color per region ID

    # plotter.add_mesh(
    #     inclined_ducted_radiator_mesh,
    #     scalars='region',            # use the 'region' array to color
    #     # categorical=True,            # treat scalars as categories
    #     show_edges=False,             # optional: show mesh edges
    #     clim=[0, 4],                 # optional: ensure color map covers all region IDs
    #     cmap=region_colors,
    #     opacity=1.0,
    # )

    plotter.add_mesh(inclined_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)  # clipped
    # =============================================================================
    # plotter.add_mesh(clipped_inclined_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)  # clipped
    # =============================================================================
    plotter.add_mesh(submesh_fan, color='red', opacity=0.4, smooth_shading=True)  # unclipped
    plotter.add_mesh(submesh_hx, color='orange', opacity=0.4, smooth_shading=True)  # unclipped
    # # =============================================================================

    plotter.view_isometric()
    plotter.camera.Azimuth(180)
    plotter.camera.Elevation(0)

    plotter.reset_camera()   # <-- prevents the zoom issue

    plotter.show(auto_close=False)
    plotter.screenshot(f"img.png", scale=3)

    import matplotlib.pyplot as plt

    img = plotter.image
    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(plotter.image)
    # ax.imshow(plotter.image[::-1, :, :])
    plt.show()

    #%% Display meshes in one window

    # plotter = pv.Plotter(
    #     off_screen=True,
    #     window_size=(1600, 1200),
    #     image_scale=3,
    # )
    # # plotter.enable_anti_aliasing("ssaa")
    # plotter.add_mesh(noflow_nacelle_mesh, show_edges=False, color='grey', opacity=0.4, smooth_shading=True)
    # plotter.add_mesh(spinner_mesh, show_edges=False, color='darkgrey', opacity=0.4, smooth_shading=True)
    # plotter.add_mesh(actuator_disk_mesh, show_edges=False, color='lightgrey', opacity=0.4, smooth_shading=True)
    # # plotter.add_mesh(radial_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.4, smooth_shading=True)  # unclipped
    # # =============================================================================
    # # plotter.add_mesh(radial_ducted_radiator_mesh, scalars='global_region', opacity=0.4, smooth_shading=True)  # unclipped
    # plotter.add_mesh(submesh_fan, color='red', opacity=0.95, smooth_shading=True)  # unclipped
    # plotter.add_mesh(submesh_hx, color='orange', opacity=0.95, smooth_shading=True)  # unclipped
    # # =============================================================================
    # plotter.add_mesh(clipped_radial_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.4, smooth_shading=True)  # clipped

    # sys.exit()

    # master_mesh = pv.merge([noflow_nacelle_mesh, spinner_mesh, inclined_ducted_radiator_mesh])

    # # Clip master mesh to negative y-values and extract slice
    # master_mesh_clip = master_mesh.clip(
    #     normal=(0,1,0),
    #     origin=(0,0,0),
    #     invert=True,
    # )
    # master_mesh_slice = master_mesh.slice(
    #     normal=(0,1,0),
    #     origin=(0,0,0),
    # )

    # p = pv.Plotter()

    # p.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.4)
    # p.add_mesh(master_mesh_slice, color="black", line_width=3)

    # p.add_axes()
    # p.show_grid()
    # p.show()
    
    return
    
    
    