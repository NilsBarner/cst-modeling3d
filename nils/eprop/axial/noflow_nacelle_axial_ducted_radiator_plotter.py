__all__ = ["plot_noflow_nacelle_axial_ducted_radiator"]

import pyvista as pv


def plot_noflow_nacelle_axial_ducted_radiator(out_dict):

    axial_ducted_radiator_mesh = out_dict["axial_ducted_radiator"]
    hx_mesh = out_dict["hx"]
    fan_mesh = out_dict["fan"]
    noflow_nacelle_mesh = out_dict["noflow_nacelle"]
    spinner_mesh = out_dict["spinner"]
    fairing_mesh = out_dict["fairing"]
    actuator_disk_mesh = out_dict["actuator_disk"]
    
    print('Hello')
    
    #%% Display meshes in one window
    
    # plotter = pv.Plotter()
    # plotter.add_mesh(noflow_nacelle_mesh, show_edges=True)
    # plotter.add_mesh(spinner_mesh, show_edges=True)
    # plotter.add_mesh(axial_ducted_radiator_mesh, show_edges=True)
    # plotter.add_mesh(actuator_disk_mesh, show_edges=True)
    # plotter.add_mesh(fairing_mesh, show_edges=True)
    # plotter.add_axes()
    # plotter.show_grid()
    # plotter.show()
    # sys.exit()
    
    #%% Display meshes in one window
    
    # mask_fan = (
    #     (axial_ducted_radiator_mesh.cell_data['global_region'] == 0) &
    #     (axial_ducted_radiator_mesh.cell_data['region'] == 3)
    # )
    # submesh_fan = axial_ducted_radiator_mesh.extract_cells(mask_fan)
    
    # mask_hx = axial_ducted_radiator_mesh.cell_data['global_region'] == 2
    # submesh_hx = axial_ducted_radiator_mesh.extract_cells(mask_hx)
    
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(1600, 1200),
        image_scale=3,
    )
    # plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(noflow_nacelle_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)
    plotter.add_mesh(spinner_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)
    plotter.add_mesh(actuator_disk_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)
    plotter.add_mesh(axial_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.2)  # , smooth_shading=True)  # unclipped
    plotter.add_mesh(fairing_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)  # unclipped
    # # =============================================================================
    # # plotter.add_mesh(radial_ducted_radiator_mesh, scalars='global_region', opacity=0.2, smooth_shading=True)  # unclipped
    # plotter.add_mesh(submesh_fan, color='red', opacity=0.95, smooth_shading=True)  # unclipped
    # plotter.add_mesh(submesh_hx, color='orange', opacity=0.95, smooth_shading=True)  # unclipped
    # # =============================================================================
    # # plotter.add_mesh(clipped_radial_ducted_radiator_mesh, show_edges=False, color='lightgrey', opacity=0.2, smooth_shading=True)  # clipped
    
    plotter.add_mesh(hx_mesh, show_edges=False, color='orange', opacity=0.2)
    plotter.add_mesh(fan_mesh, show_edges=False, color='red', opacity=0.2)
    
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
    plt.show()
    
    #%%
    
    engine = pv.merge([noflow_nacelle_mesh, spinner_mesh, actuator_disk_mesh, axial_ducted_radiator_mesh, fairing_mesh])
    
    # =============================================================================
    nac_origin = [[14.89132157, 6.11212032, -1.7223025 ]]
    
    engine.points[:, 0] += 14.89132157
    engine.points[:, 1] += 6.11212032
    engine.points[:, 2] += -1.7223025
    # =============================================================================
    
    # Load airframe STL
    # airframe = pv.read(r"C:\Users\nmb48\Documents\GitHub\SUAVE\nils\openvsp_gmsh_su2\tasopt\base.stl")
    airframe = pv.read(r"C:\Users\nmb48\Documents\GitHub\SUAVE\base.stl")
    
    # You already have:
    # engine = your PolyData object
    
    plotter = pv.Plotter()
    
    plotter.add_mesh(airframe, color='lightgrey', opacity=1.0)
    plotter.add_mesh(engine, color='red')
    
    plotter.show()
    
    return


