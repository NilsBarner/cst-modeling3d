import numpy as np

def sweep_profiles_around_curve(
    profiles,
    section_r,
    section_theta_deg,
    section_x=None,
    section_scale=None,
    n_spanwise=51,
    interp_kind='linear',
):
    """
    Sweep a list of 2D profiles around a (r,theta) closed guide curve.

    Parameters
    ----------
    profiles : List[List[np.ndarray]]
        List of [profile_x, profile_y] arrays (profile_x, profile_y are 1D, same length for all profiles).
        length = n_sections
    section_r : array_like
        radius for each section reference point (length == n_sections)
    section_theta_deg : array_like
        angle in degrees for each section reference point (length == n_sections)
    section_x : None or array_like or scalar
        axial offset for each section (length==n_sections or scalar). Defaults to zeros.
    section_scale : None or array_like or scalar
        scale factor for each section (length==n_sections or scalar). Defaults to ones.
    n_spanwise : int
        number of circumferential samples per surface block (same as your previous n_spanwise).
    interp_kind : {'linear','cubic'} (cubic uses scipy if available)
        interpolation method along the circumferential (periodic) direction for each profile point.

    Returns
    -------
    surfs : List[List[np.ndarray]]
        List of [surf_x, surf_y, surf_z] for each profile-block (len = n_sections).
        Each surf_* shape is (n_spanwise, n_point).
    """

    # basic checks
    n_sections = len(profiles)
    if n_sections < 2:
        raise ValueError("Need at least two profiles to sweep.")
    profile_point_count = profiles[0][0].shape[0]
    for pr in profiles:
        if pr[0].shape[0] != profile_point_count or pr[1].shape[0] != profile_point_count:
            raise ValueError("All profiles must have the same number of points.")

    section_r = np.asarray(section_r, dtype=float)
    section_theta_deg = np.asarray(section_theta_deg, dtype=float)

    if section_r.shape[0] != n_sections or section_theta_deg.shape[0] != n_sections:
        raise ValueError("section_r and section_theta_deg must have length == len(profiles).")

    # default section_x and section_scale
    if section_x is None:
        section_x = np.zeros(n_sections, dtype=float)
    if section_scale is None:
        section_scale = np.ones(n_sections, dtype=float)

    # allow scalars
    if np.isscalar(section_x):
        section_x = np.ones(n_sections, dtype=float) * float(section_x)
    else:
        section_x = np.asarray(section_x, dtype=float)
    if np.isscalar(section_scale):
        section_scale = np.ones(n_sections, dtype=float) * float(section_scale)
    else:
        section_scale = np.asarray(section_scale, dtype=float)

    # build parametric control coordinate for each section (periodic)
    control_s = np.linspace(0.0, 1.0, n_sections, endpoint=False)  # e.g. 0, 1/n, 2/n, ...
    # global s for the full guide curve (total samples)
    n_total = n_sections * (n_spanwise - 1) + 1
    global_s = np.linspace(0.0, 1.0, n_total)

    # Prepare per-span radius, angle, x, scale by periodic interpolation of the control arrays
    # We will handle wrapping by appending the first value at 1.0 (periodic)
    def periodic_interp(control_s, control_vals, query_s, kind='linear'):
        # control_s assumed sorted ascending in [0,1)
        # make arrays for periodic interpolation
        cs = np.concatenate((control_s, control_s[:1] + 1.0))
        cv = np.concatenate((control_vals, control_vals[:1]))
        if kind == 'linear' or kind == 'cubic' and not _HAS_CUBIC:
            # use numpy.interp (linear) on query; ensure query_s in ascending order 0..1
            qs = np.array(query_s)
            # any query values < cs[0] are fine; np.interp requires x ascending
            # we must shift query values that are < cs[0] by +1 if cs[0] > 0 (not the case here)
            # but because cs extends to >1 we can safely call interp
            return np.interp(qs, cs, cv)
        else:
            # cubic using SciPy's CubicSpline with periodic BC
            from scipy.interpolate import CubicSpline
            # construct spline on control_s extended with +1 duplicate
            cs0 = control_s
            cv0 = control_vals
            # CubicSpline with bc_type='periodic' expects strictly periodic data; we'll create with control_s sorted
            # build spline on control_s and extend periodicity implicitly by bc_type
            spline = CubicSpline(control_s, control_vals, bc_type='periodic')
            return spline(query_s % 1.0)

    # optional: use cubic interpolation if scipy available
    try:
        from scipy.interpolate import CubicSpline  # noqa: F401
        _HAS_CUBIC = True
    except Exception:
        _HAS_CUBIC = False

    # compute per-span placement parameters
    radii_full = periodic_interp(control_s, section_r, global_s, kind='cubic' if (interp_kind == 'cubic' and _HAS_CUBIC) else 'linear')
    theta_full = periodic_interp(control_s, section_theta_deg, global_s, kind='cubic' if (interp_kind == 'cubic' and _HAS_CUBIC) else 'linear')
    x_full = periodic_interp(control_s, section_x, global_s, kind='linear')
    scale_full = periodic_interp(control_s, section_scale, global_s, kind='linear')

    # convert theta to radians for trig
    theta_rad = np.deg2rad(theta_full)

    # compute global reference positions (y,z) from r,theta
    y_full = radii_full * np.cos(theta_rad)
    z_full = radii_full * np.sin(theta_rad)

    # build spanwise_profiles: for every profile point index we interpolate the profile control values around sections
    # spanwise_profiles will be a list of length n_total, each entry [px, py] arrays length profile_point_count
    spanwise_profiles = [[np.zeros(profile_point_count, dtype=float), np.zeros(profile_point_count, dtype=float)] for _ in range(n_total)]

    # For each profile point (index over profile curve coords) we have control values = profiles[i_section][0 or 1][i_point]
    # We will perform periodic interpolation for each point
    for i_point in range(profile_point_count):
        control_x_vals = np.array([profiles[i_s][0][i_point] for i_s in range(n_sections)])
        control_y_vals = np.array([profiles[i_s][1][i_point] for i_s in range(n_sections)])

        interp_kind_effective = interp_kind if (interp_kind != 'cubic' or _HAS_CUBIC) else 'linear'

        xx = periodic_interp(control_s, control_x_vals, global_s, kind=interp_kind_effective)
        yy = periodic_interp(control_s, control_y_vals, global_s, kind=interp_kind_effective)

        for i_span in range(n_total):
            spanwise_profiles[i_span][0][i_point] = xx[i_span]
            spanwise_profiles[i_span][1][i_point] = yy[i_span]

    # local transform: rotate profile_y about x-axis by rot (theta_rad) and translate by x_full, y_full, z_full
    def transform_profile(profile_x, profile_y, x0, y0, z0, scale, rot_rad):
        # X = x0 + scale * profile_x
        # Y = y0 + scale * profile_y * cos(rot)
        # Z = z0 + scale * profile_y * sin(rot)
        sx = scale
        X = x0 + profile_x * sx
        Y = y0 + profile_y * np.cos(rot_rad) * sx
        Z = z0 + profile_y * np.sin(rot_rad) * sx
        return X, Y, Z

    # Make the surfaces (there are n_sections surface blocks, each using n_spanwise rows)
    surfs = []
    for i_block in range(n_sections):
        surf_x = np.zeros((n_spanwise, profile_point_count), dtype=float)
        surf_y = np.zeros((n_spanwise, profile_point_count), dtype=float)
        surf_z = np.zeros((n_spanwise, profile_point_count), dtype=float)

        for i_local in range(n_spanwise):
            i_total = i_block * (n_spanwise - 1) + i_local
            px = spanwise_profiles[i_total][0]
            py = spanwise_profiles[i_total][1]
            x0 = float(x_full[i_total])
            y0 = float(y_full[i_total])
            z0 = float(z_full[i_total])
            scale = float(scale_full[i_total])
            rot = float(theta_rad[i_total])
            X, Y, Z = transform_profile(px, py, x0, y0, z0, scale, rot)
            surf_x[i_local, :] = X
            surf_y[i_local, :] = Y
            surf_z[i_local, :] = Z

        surfs.append([surf_x, surf_y, surf_z])

    return surfs