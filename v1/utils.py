import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
import scipy.interpolate


laserFoV = np.radians(225)


def laser_angles(N, fov=None):
    fov = fov or laserFoV
    return np.linspace(-fov*0.5, fov*0.5, N)


def xy_to_rphi(x, y):
    # NOTE: Axes rotated by 90 CCW by intent, so tat 0 is top.
    return np.hypot(x, y), np.arctan2(-x, y)


def rphi_to_xy(r, phi):
    return r * -np.sin(phi), r * np.cos(phi)


def scan_to_xy(scan, thresh=None, fov=None):
    s = np.array(scan, copy=True)
    if thresh is not None:
        s[s > thresh] = np.nan
    return rphi_to_xy(s, laser_angles(len(scan), fov))


def load_scan(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, scans = data[:,0].astype(np.uint32), data[:,1:-1]
    return seqs, scans


def load_dets(name):
    def _doload(fname):
        seqs, dets = [], []
        with open(fname) as f:
            for line in f:
                seq, tail = line.split(',', 1)
                seqs.append(int(seq))
                dets.append(json.loads(tail))
        return seqs, dets

    s1, wcs = _doload(name + ".wc")
    s2, was = _doload(name + ".wa")

    assert all(a == b for a, b in zip(s1, s2)), "Uhhhh?"
    return s1, wcs, was


def precrec_unvoted(preds, gts, radius, pred_rphi=False, gt_rphi=False):
    """
    The "unvoted" precision/recall, meaning that multiple predictions for the same ground-truth are NOT penalized.

    - `preds` an iterable (scans) of iterables (per scan) containing predicted x/y or r/phi pairs.
    - `gts` an iterable (scans) of iterables (per scan) containing ground-truth x/y or r/phi pairs.
    - `radius` the cutoff-radius for "correct", in meters.
    - `pred_rphi` whether `preds` is r/phi (True) or x/y (False).
    - `gt_rphi` whether `gts` is r/phi (True) or x/y (False).

    Returns a pair of numbers: (precision, recall)
    """
    # Tested against other code.

    npred, npred_hit, ngt, ngt_hit = 0.0, 0.0, 0.0, 0.0
    for ps, gts in zip(preds, gts):
        # Distance between each ground-truth and predictions
        assoc = np.zeros((len(gts), len(ps)))

        for ip, p in enumerate(ps):
            for igt, gt in enumerate(gts):
                px, py = rphi_to_xy(*p) if pred_rphi else p
                gx, gy = rphi_to_xy(*gt) if gt_rphi else gt
                assoc[igt, ip] = np.hypot(px-gx, py-gy)

        # Now cutting it off at `radius`, we can get all we need.
        assoc = assoc < radius
        npred += len(ps)
        npred_hit += np.count_nonzero(np.sum(assoc, axis=0))
        ngt += len(gts)
        ngt_hit += np.count_nonzero(np.sum(assoc, axis=1))

    return (
        npred_hit/npred if npred > 0 else np.nan,
          ngt_hit/ngt   if   ngt > 0 else np.nan
    )


def precrec(preds, gts, radius, pred_rphi=False, gt_rphi=False):
    """
    Ideally, we'd use Hungarian algorithm instead of greedy one on all "hits" within the radius, but meh.

    - `preds` an iterable (scans) of iterables (per scan) containing predicted x/y or r/phi pairs.
    - `gts` an iterable (scans) of iterables (per scan) containing ground-truth x/y or r/phi pairs.
    - `radius` the cutoff-radius for "correct", in meters.
    - `pred_rphi` whether `preds` is r/phi (True) or x/y (False).
    - `gt_rphi` whether `gts` is r/phi (True) or x/y (False).

    Returns a pair of numbers: (precision, recall)
    """
    tp, fp, fn = 0.0, 0.0, 0.0
    for ps, gts in zip(preds, gts):
        # Assign each ground-truth the prediction which is closest to it AND inside the radius.
        assoc = np.zeros((len(gts), len(ps)))
        for igt, gt in enumerate(gts):
            min_d = radius
            best = -1
            for ip, p in enumerate(ps):
                # Skip prediction if already associated.
                if np.any(assoc[:,ip]):
                    continue

                px, py = rphi_to_xy(*p) if pred_rphi else p
                gx, gy = rphi_to_xy(*gt) if gt_rphi else gt
                d = np.hypot(px-gx, py-gy)
                if d < min_d:
                    min_d = d
                    best = ip

            if best != -1:
                assoc[igt,best] = 1

        nassoc = np.sum(assoc)
        tp += nassoc  # All associated predictions are true pos.
        fp += len(ps) - nassoc  # All not-associated predictions are false pos.
        fn += len(gts) - nassoc  # All not-associated ground-truths are false negs.

    return tp/(fp+tp) if fp+tp > 0 else np.nan, tp/(fn+tp) if fn+tp > 0 else np.nan


# Tested with gts,gts -> 1,1 and the following -> (0.5, 0.6666)
# precrec(
#    preds=[[(-1,0),(0,0),(1,0),(0,1)]],
#    gts=[[(-0.5,0),(0.5,0),(-2,-2)]],
#    radius=0.6
# )


def prettify_pr_curve(ax):
    ax.plot([0,1], [0,1], ls="--", c=".6")
    ax.set_xlim(-0.02,1.02)
    ax.set_ylim(-0.02,1.02)
    ax.set_xlabel("Recall [%]")
    ax.set_ylabel("Precision [%]")
    ax.axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*100)))
    ax.axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*100)))
    return ax


def votes_to_detections(locations, probas=None, in_rphi=True, out_rphi=True, bin_size=0.1, blur_win=21, blur_sigma=2.0, x_min=-15.0, x_max=15.0, y_min=-5.0, y_max=15.0, retgrid=False):
    '''
    Convert a list of votes to a list of detections based on Non-Max supression.

    - `locations` an iterable containing predicted x/y or r/phi pairs.
    - `probas` an iterable containing predicted probabilities. Considered all ones if `None`.
    - `in_rphi` whether `locations` is r/phi (True) or x/y (False).
    - `out_rphi` whether the output should be r/phi (True) or x/y (False).
    - `bin_size` the bin size (in meters) used for the grid where votes are cast.
    - `blur_win` the window size (in bins) used to blur the voting grid.
    - `blur_sigma` the sigma used to compute the Gaussian in the blur window.
    - `x_min` the left limit for the voting grid, in meters.
    - `x_max` the right limit for the voting grid, in meters.
    - `y_min` the bottom limit for the voting grid in meters.
    - `y_max` the top limit for the voting grid in meters.

    Returns a list of tuples (x,y,class) or (r,phi,class) where `class` is
    the index into `probas` which was highest for each detection, thus starts at 0.

    NOTE/TODO: We really should replace `bin_size` by `nbins` so as to avoid "remainders".
               Right now, we simply ignore the remainder on the "max" side.
    '''
    locations = np.array(locations)
    if len(locations) == 0:
        return []

    if probas is None:
        probas = np.ones((len(locations),1))
    else:
        probas = np.array(probas)
        assert len(probas) == len(locations) and probas.ndim == 2, "Invalid format of `probas`"

    x_range = int((x_max-x_min)/bin_size)
    y_range = int((y_max-y_min)/bin_size)
    grid = np.zeros((x_range, y_range, 1+probas.shape[1]), np.float32)

    # Update x/y max to correspond to the end of the last bin.
    # TODO: fix this as stated in the docstring.
    x_max = x_min + x_range*bin_size
    y_max = y_min + y_range*bin_size

    # Do the voting into the grid.
    for loc, p in zip(locations, probas):
        x,y = rphi_to_xy(*loc) if in_rphi else loc

        # Skip votes outside the grid.
        if not (x_min < x < x_max and y_min < y < y_max):
            continue

        x = int((x-x_min)/bin_size)
        y = int((y-y_min)/bin_size)
        grid[x,y,0] += np.sum(p)
        grid[x,y,1:] += p

    # Yes, this blurs each channel individually, just what we need!
    grid = cv2.GaussianBlur(grid, (blur_win,blur_win), blur_sigma)

    # Find the maxima (NMS) only in the "common" voting grid.
    grid_all = grid[:,:,0]
    max_grid = scipy.ndimage.maximum_filter(grid_all, size=3)
    maxima = (grid_all == max_grid) & (grid_all != 0)
    m_x, m_y = np.where(maxima)

    # Probabilities of all classes where maxima were found.
    m_p = grid[m_x, m_y, 1:]

    # Back from grid-bins to real-world locations.
    m_x = m_x*bin_size + x_min + bin_size/2
    m_y = m_y*bin_size + y_min + bin_size/2
    maxima = [(xy_to_rphi(x,y) if out_rphi else (x,y)) + (np.argmax(p),) for x,y,p in zip(m_x, m_y, m_p)]
    return (maxima, grid) if retgrid else maxima


def generate_cut_outs(scan, standard_depth=4.0, window_size=48, threshold_distance=1.0, npts=None, center=True, border=29.99, resample_type='cv', **kw):
    '''
    Generate window cut outs that all have a fixed size independent of depth.
    This means areas close to the scanner will be subsampled and areas far away
    will be upsampled.
    All cut outs will have values between `-threshold_distance` and `+threshold_distance`
    as they are normalized by the center point.

    - `scan` an iterable of radii within a laser scan.
    - `standard_depth` the reference distance (in meters) at which a window with `window_size` gets cut out.
    - `window_size` the window of laser rays that will be extracted everywhere.
    - `npts` is the number of final samples to have per window. `None` means same as `window_size`.
    - `threshold_distance` the distance in meters from the center point that will be used to clamp the laser radii.
      Since we're talking about laser-radii, this means the cutout is a donut-shaped hull, as opposed to a rectangular hull.
      This can be `np.inf` to skip the clamping altogether.
    - `center` whether to center the cutout around the current laser point's depth (True), or keep depth values raw (False).
    - `border` the radius value to fill the half of the outermost windows with.
    - `resample_type` specifies the resampling API to be used. Possible values are:
        - `cv` for OpenCV's `cv2.resize` function using LINEAR/AREA interpolation.
        - `zoom` for SciPy's `zoom` function, to which options such as `order=3` can be passed as extra kwargs.
        - `int1d` for SciPy's `interp1d` function, to which options such as `kind=3` can be passed as extra kwargs.
    '''
    s_np = np.fromiter(iter(scan), dtype=np.float32)
    N = len(s_np)

    npts = npts or window_size
    cut_outs = np.zeros((N, npts), dtype=np.float32)

    current_size = (window_size * standard_depth / s_np).astype(np.int32)
    start = -current_size//2 + np.arange(N)
    end = start + current_size
    s_np_extended = np.append(s_np, border)

    # While we don't really need to special-case, it should save precious computation.
    if threshold_distance != np.inf:
        near = s_np-threshold_distance
        far  = s_np+threshold_distance

    for i in range(N):
        # Get the window.
        sample_points = np.arange(start[i], end[i])
        sample_points[sample_points < 0] = -1
        sample_points[sample_points >= N] = -1
        window = s_np_extended[sample_points]

        # Threshold the near and far values, then
        if threshold_distance != np.inf:
            window = np.clip(window, near[i], far[i])

        # shift everything to be centered around the middle point.
        if center:
            window -= s_np[i]

        # Values will now span [-d,d] if `center` and `clamp` are both True.

        # resample it to the correct size.
        if resample_type == 'cv':
            # Use 'INTER_LINEAR' for when down-sampling the image LINEAR is ridiculous.
            # It's just 1ms slower for a whole scan in the worst case.
            interp = cv2.INTER_AREA if npts < len(window) else cv2.INTER_LINEAR
            cut_outs[i,:] = cv2.resize(window[None], (npts,1), interpolation=interp)[0]
        elif resample_type == 'zoom':
            scipy.ndimage.interpolation.zoom(window, npts/len(window), output=cut_outs[i,:], **kw)
        elif resample_type == 'int1d':
            cut_outs[i,:] = scipy.interpolate.interp1d(np.linspace(0,1, num=len(window), endpoint=True), window, assume_sorted=True, copy=False, **kw)(np.linspace(0,1,num=npts, endpoint=True))

    return cut_outs


def generate_cut_outs_raw(scan, window_size=48, threshold_distance=np.inf, center=False, border=29.99):
    '''
    Generate window cut outs that all have a fixed number of rays independent of depth.
    This means objects close to the scanner will cover more rays and those far away fewer.
    All cut outs will contain the raw values from the input scan.

    - `scan` an iterable of radii within a laser scan.
    - `window_size` the window of laser rays that will be extracted everywhere.
    - `threshold_distance` the distance in meters from the center point that will be used to clamp the laser radii.
      Since we're talking about laser-radii, this means the cutout is a donut-shaped hull, as opposed to a rectangular hull.
      This can be `np.inf` to skip the clamping altogether.
    - `center` whether to center the cutout around the current laser point's depth (True), or keep depth values raw (False).
    - `border` the radius value to fill the half of the outermost windows with.
    '''
    s_np = np.fromiter(iter(scan), dtype=np.float32)
    N = len(s_np)

    cut_outs = np.zeros((N, window_size), dtype=np.float32)

    start = -window_size//2 + np.arange(N)
    end = start + window_size
    s_np_extended = np.append(s_np, border)

    # While we don't really need to special-case, it should save precious computation.
    if threshold_distance != np.inf:
        near = s_np-threshold_distance
        far  = s_np+threshold_distance

    for i in range(N):
        # Get the window.
        sample_points = np.arange(start[i], end[i])
        sample_points[sample_points < 0] = -1
        sample_points[sample_points >= N] = -1
        window = s_np_extended[sample_points]

        # Threshold the near and far values, then
        if threshold_distance != np.inf:
            window = np.clip(window, near[i], far[i])

        # shift everything to be centered around the middle point.
        if center:
            window -= s_np[i]

        cut_outs[i,:] = window

    return cut_outs


def hyperopt(pred_conf):
    ho_wBG = 0.38395839618267696
    ho_wWC = 0.599481486880304
    ho_wWA = 0.4885948464627302

    # Unused
    ho_sigma = 2.93
    ho_binsz = 0.10

    # Compute "optimal" "tight" window-size dependent on blur-size.
    ho_blur_win = ho_sigma*5
    ho_blur_win = int(2*(ho_blur_win//2)+1)  # Make odd

    # Weight network outputs
    newconf = pred_conf * [ho_wBG, ho_wWC, ho_wWA]
    # And re-normalize to get "real" probabilities
    newconf /= np.sum(newconf, axis=-1, keepdims=True)

    return newconf, {'bin_size': ho_binsz, 'blur_win': ho_blur_win, 'blur_sigma': ho_sigma}
