from collections import defaultdict
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
import scipy.interpolate
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc

import lbtoolbox.plotting as lbplt


laserIncrement = np.radians(0.5)
laserFoV = (450-1)*laserIncrement  # 450 points.


def laser_angles(N, fov=None):
    fov = fov or laserFoV
    return np.linspace(-fov*0.5, fov*0.5, N)


def xy_to_rphi(x, y):
    # NOTE: Axes rotated by 90 CCW by intent, so that 0 is top.
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
    seqs, times, scans = data[:,0].astype(np.uint32), data[:,1].astype(np.float32), data[:,2:].astype(np.float32)
    return seqs, times, scans


def load_odom(fname):
    return np.genfromtxt(fname, delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])


def load_dets(name, DATADIR, LABELDIR):
    def _doload(fname):
        seqs, dets = [], []
        with open(fname) as f:
            for line in f:
                seq, tail = line.split(',', 1)
                seqs.append(int(seq))
                dets.append(json.loads(tail))
        return seqs, dets

    s1, wcs = _doload(name.replace(DATADIR, LABELDIR) + '.wc')
    s2, was = _doload(name.replace(DATADIR, LABELDIR) + '.wa')
    s3, wps = _doload(name.replace(DATADIR, LABELDIR) + '.wp')

    assert all(a == b == c for a, b, c in zip(s1, s2, s3)), "Uhhhh?"
    return np.array(s1), wcs, was, wps


def linearize(all_seqs, all_scans, all_detseqs, all_wcs, all_was, all_wps):
    lin_seqs, lin_scans, lin_wcs, lin_was, lin_wps = [], [], [], [], []
    # Loop through the "sessions" (correspond to files)
    for seqs, scans, detseqs, wcs, was, wps in zip(all_seqs, all_scans, all_detseqs, all_wcs, all_was, all_wps):
        # Note that sequence IDs may overlap between sessions!
        s2s = dict(zip(seqs, scans))
        # Go over the individual measurements/annotations of a session.
        for ds, wc, wa, wp in zip(detseqs, wcs, was, wps):
            lin_seqs.append(ds)
            lin_scans.append(s2s[ds])
            lin_wcs.append(wc)
            lin_was.append(wa)
            lin_wps.append(wp)
    return lin_seqs, lin_scans, lin_wcs, lin_was, lin_wps


def closest_detection(scan, dets, radii):
    """
    Given a single `scan` (450 floats), a list of r,phi detections `dets` (Nx2),
    and a list of N `radii` for those detections, return a mapping from each
    point in `scan` to the closest detection for which the point falls inside its radius.
    The returned detection-index is a 1-based index, with 0 meaning no detection
    is close enough to that point.
    """
    if len(dets) == 0:
        return np.zeros_like(scan, dtype=int)

    assert len(dets) == len(radii), "Need to give a radius for each detection!"

    scan_xy = np.array(scan_to_xy(scan)).T

    # Distance (in x,y space) of each laser-point with each detection.
    dists = cdist(scan_xy, np.array([rphi_to_xy(x,y) for x,y in dets]))

    # Subtract the radius from the distances, such that they are <0 if inside, >0 if outside.
    dists -= radii

    # Prepend zeros so that argmin is 0 for everything "outside".
    dists = np.hstack([np.zeros((len(scan),1)), dists])

    # And find out who's closest, including the threshold!
    return np.argmin(dists, axis=1)


def global2win(r, phi, dr, dphi):
    # Convert to relative, angle-aligned x/y coordinate-system.
    dx = np.sin(dphi-phi) * dr
    dy = np.cos(dphi-phi) * dr - r
    return dx, dy


def win2global(r, phi, dx, dy):
    y = r + dy
    dphi = np.arctan2(dx, y)  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    return y/np.cos(dphi), phi + dphi


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


def votes_to_detections(locations, probas=None, in_rphi=True, out_rphi=True,
                        bin_size=0.1, blur_win=21, blur_sigma=2.0,
                        x_min=-15.0, x_max=15.0, y_min=-5.0, y_max=15.0, retgrid=False):
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
        return ([], 0) if retgrid else []

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


def generate_cut_outs(scan, standard_depth=4.0, window_size=48, threshold_distance=1.0, npts=None,
                      center='point', border=29.99, resample_type='cv', **kw):
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
    - `center` Defines how the cutout depth-values will be centered/rescaled:
        - 'none'/'raw'/None: keep the raw depth values in meters.
        - 'point': move the depth values such that the center-point of the cutout is at zero.
        - 'near': move the depth values such that the 'near' cutoff is at zero (it's like 'point' + `threshold_distance`).
        - 'far': actually turn depth upside down, such that the 'far' threshold is at zero,
                 the current laser point at `threshold_distance` and the 'near' at `2*threshold_distance`.
                 This, combined with `border=29.99` has the advantage of putting border to zero.
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
        if center == 'point':
            window -= s_np[i]
        elif center == 'near' and threshold_distance != np.inf:
            window -= near[i]
        elif center == 'far' and threshold_distance != np.inf:
            window = far[i] - window
        elif center not in (None, 'none', 'raw'):
            raise ValueError("unknown `center` parameter " + str(center))

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


########
# New eval
########


def vote_avg(vx, vy, p):
    return np.mean(vx), np.mean(vy), np.mean(p, axis=0)


def agnostic_weighted_vote_avg(vx, vy, p):
    weights = np.sum(p[:,1:], axis=1)
    norm = 1.0/np.sum(weights)
    return norm*np.sum(weights*vx), norm*np.sum(weights*vy), norm*np.sum(weights[:,None]*p, axis=0)


def max_weighted_vote_avg(vx, vy, p):
    weights = np.max(p[:,1:], axis=1)
    norm = 1.0/np.sum(weights)
    return norm*np.sum(weights*vx), norm*np.sum(weights*vy), norm*np.sum(weights[:,None]*p, axis=0)


def votes_to_detections2(xs, ys, probas, weighted_avg=False, min_thresh=1e-5,
                         bin_size=0.1, blur_win=21, blur_sigma=2.0,
                         x_min=-15.0, x_max=15.0, y_min=-5.0, y_max=15.0,
                         vote_collect_radius=0.3, retgrid=False,
                         class_weights=None):
    '''
    Convert a list of votes to a list of detections based on Non-Max suppression.

    ` `vote_combiner` the combination function for the votes per detection.
    - `bin_size` the bin size (in meters) used for the grid where votes are cast.
    - `blur_win` the window size (in bins) used to blur the voting grid.
    - `blur_sigma` the sigma used to compute the Gaussian in the blur window.
    - `x_min` the left limit for the voting grid, in meters.
    - `x_max` the right limit for the voting grid, in meters.
    - `y_min` the bottom limit for the voting grid in meters.
    - `y_max` the top limit for the voting grid in meters.
    - `vote_collect_radius` the radius use during the collection of votes assigned
      to each detection.

    Returns a list of tuples (x,y,probs) where `probs` has the same layout as
    `probas`.
    '''
    if class_weights is not None:
        probas = np.array(probas)  # Make a copy.
        probas[:,:,1:] *= class_weights
    vote_combiner = agnostic_weighted_vote_avg if weighted_avg is True else vote_avg
    x_range = int((x_max-x_min)/bin_size)
    y_range = int((y_max-y_min)/bin_size)
    grid = np.zeros((x_range, y_range, probas.shape[2]), np.float32)

    vote_collect_radius_sq = vote_collect_radius * vote_collect_radius

    # Update x/y max to correspond to the end of the last bin.
    x_max = x_min + x_range*bin_size
    y_max = y_min + y_range*bin_size

    # Where we collect the outputs.
    all_dets = []
    all_grids = []

    # Iterate over the scans. TODO: We can do most of this outside the looping too, actually.
    for iscan, (x, y, probs) in enumerate(zip(xs, ys, probas)):
        # Clear the grid, for each scan its own.
        grid.fill(0)
        all_dets.append([])

        # Filter out all the super-weak votes, as they wouldn't contribute much anyways
        # but waste time.
        voters_idxs = np.where(np.sum(probs[:,1:], axis=-1) > min_thresh)[0]
        # voters_idxs = np.where(probs[:,0] < 1-min_thresh)[0]
        # voters_idxs = np.where(np.any(probs[:,1:] > min_thresh, axis=-1))[0]

        # No voters, early bail
        if not len(voters_idxs):
            if retgrid:
                all_grids.append(np.array(grid))  # Be sure to make a copy.
            continue

        x = x[voters_idxs]
        y = y[voters_idxs]
        probs = probs[voters_idxs]

        # Convert x/y to grid-cells.
        x_idx = np.int64((x-x_min)/bin_size)
        y_idx = np.int64((y-y_min)/bin_size)

        # Discard data outside of the window.
        mask = (0 <= x_idx) & (x_idx < x_range) & (0 <= y_idx) & (y_idx < y_range)
        x_idx = x_idx[mask]
        x = x[mask]
        y_idx = y_idx[mask]
        y = y[mask]
        probs = probs[mask]

        # Vote into the grid, including the agnostic vote as sum of class-votes!
        #TODO Do we need the class grids?
        np.add.at(grid, [x_idx, y_idx], np.concatenate([np.sum(probs[:,1:], axis=-1, keepdims=True), probs[:,1:]], axis=-1))

        # Find the maxima (NMS) only in the "common" voting grid.
        grid_all = grid[:,:,0]
        if blur_win is not None and blur_sigma is not None:
            grid_all = cv2.GaussianBlur(grid_all, (blur_win,blur_win), blur_sigma)
        max_grid = scipy.ndimage.maximum_filter(grid_all, size=3)
        maxima = (grid_all == max_grid) & (grid_all > 0)
        m_x, m_y = np.where(maxima)

        if len(m_x) == 0:
            if retgrid:
                all_grids.append(np.array(grid))  # Be sure to make a copy.
            continue

        # Back from grid-bins to real-world locations.
        m_x = m_x*bin_size + x_min + bin_size/2
        m_y = m_y*bin_size + y_min + bin_size/2

        # For each vote, get which maximum/detection it contributed to.
        # Shape of `center_dist` (ndets, voters) and outer is (voters)
        center_dist = np.square(x - m_x[:,None]) + np.square(y - m_y[:,None])
        det_voters = np.argmin(center_dist, axis=0)

        # Generate the final detections by average over their voters.
        for ipeak in range(len(m_x)):
            my_voter_idxs = np.where(det_voters == ipeak)[0]
            my_voter_idxs = my_voter_idxs[center_dist[ipeak, my_voter_idxs] < vote_collect_radius_sq]
            all_dets[-1].append(vote_combiner(x[my_voter_idxs], y[my_voter_idxs], probs[my_voter_idxs,:]))

        if retgrid:
            all_grids.append(np.array(grid))  # Be sure to make a copy.

    if retgrid:
        return all_dets, all_grids
    return all_dets


# Convert it to flat `x`, `y`, `probs` arrays and an extra `frame` array,
# which is the index they had in the first place.
def deep2flat(dets):
    all_x, all_y, all_p, all_frames = [], [], [], []
    for i, ds in enumerate(dets):
        for (x, y, p) in ds:
            all_x.append(x)
            all_y.append(y)
            all_p.append(p)
            all_frames.append(i)
    return np.array(all_x), np.array(all_y), np.array(all_p), np.array(all_frames)


# Same but slightly different for the ground-truth.
def deep2flat_gt(gts, radius):
    all_x, all_y, all_r, all_frames = [], [], [], []
    for i, gt in enumerate(gts):
        for (r, phi) in gt:
            x, y = rphi_to_xy(r, phi)
            all_x.append(x)
            all_y.append(y)
            all_r.append(radius)
            all_frames.append(i)
    return np.array(all_x), np.array(all_y), np.array(all_r), np.array(all_frames)


def prec_rec_2d(det_scores, det_coords, det_frames, gt_coords, gt_frames, gt_radii):
    """ Computes full precision-recall curves at all possible thresholds.

    Arguments:
    - `det_scores` (D,) array containing the scores of the D detections.
    - `det_coords` (D,2) array containing the (x,y) coordinates of the D detections.
    - `det_frames` (D,) array containing the frame number of each of the D detections.
    - `gt_coords` (L,2) array containing the (x,y) coordinates of the L labels (ground-truth detections).
    - `gt_frames` (L,) array containing the frame number of each of the L labels.
    - `gt_radii` (L,) array containing the radius at which each of the L labels should consider detection associations.
                      This will typically just be an np.full_like(gt_frames, 0.5) or similar,
                      but could vary when mixing classes, for example.

    Returns: (recs, precs, threshs)
    - `threshs`: (D,) array of sorted thresholds (scores), from higher to lower.
    - `recs`: (D,) array of recall scores corresponding to the thresholds.
    - `precs`: (D,) array of precision scores corresponding to the thresholds.
    """
    # This means that all reported detection frames which are not in ground-truth frames
    # will be counted as false-positives.
    # TODO: do some sanity-checks in the "linearization" functions before calling `prec_rec_2d`.
    frames = np.unique(np.r_[det_frames, gt_frames])

    det_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(frames), dtype=np.uint32)
    fps = np.zeros(len(frames), dtype=np.uint32)
    fns = np.array([np.sum(gt_frames == f) for f in frames], dtype=np.uint32)

    precs = np.full_like(det_scores, np.nan)
    recs = np.full_like(det_scores, np.nan)
    threshs = np.full_like(det_scores, np.nan)

    indices = np.argsort(det_scores, kind='mergesort')  # mergesort for determinism.
    for i, idx in enumerate(reversed(indices)):
        frame = det_frames[idx]
        iframe = np.where(frames == frame)[0][0]  # Can only be a single one.

        # Accept this detection
        dets_idxs = det_accepted_idxs[frame]
        dets_idxs.append(idx)
        threshs[i] = det_scores[idx]

        dets = det_coords[dets_idxs]

        gts_mask = gt_frames == frame
        gts = gt_coords[gts_mask]
        radii = gt_radii[gts_mask]

        if len(gts) == 0:  # No GT, but there is a detection.
            fps[iframe] += 1
        else:              # There is GT and detection in this frame.
            not_in_radius = radii[:,None] < cdist(gts, dets)  # -> ngts x ndets, True (=1) if too far, False (=0) if may match.
            igt, idet = linear_sum_assignment(not_in_radius)

            tps[iframe] = np.sum(np.logical_not(not_in_radius[igt, idet]))  # Could match within radius
            fps[iframe] = len(dets) - tps[iframe]  # NB: dets is only the so-far accepted.
            fns[iframe] = len(gts) - tps[iframe]

        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precs[i] = tp/(fp+tp) if fp+tp > 0 else np.nan
        recs[i] = tp/(fn+tp) if fn+tp > 0 else np.nan

    return recs, precs, threshs


def _prepare_prec_rec_softmax(scans, pred_offs):
    angles = laser_angles(scans.shape[-1])[None,:]
    return rphi_to_xy(*win2global(scans, angles, pred_offs[:,:,0], pred_offs[:,:,1]))

def _prepare_prec_rec_sigmoids(scans, pred_offs, pred_conf):
    angles = laser_angles(scans.shape[-1])[None,:]
    x1, y1 = rphi_to_xy(*win2global(scans, angles, pred_offs[:,:,0], pred_offs[:,:,1]))
    x2, y2 = rphi_to_xy(*win2global(scans, angles, pred_offs[:,:,2], pred_offs[:,:,3]))
    x3, y3 = rphi_to_xy(*win2global(scans, angles, pred_offs[:,:,4], pred_offs[:,:,5]))
    x = np.c_[x1, x2, x3]
    y = np.c_[y1, y2, y3]
    zero = np.zeros_like(pred_conf[:,:,1])
    pred_conf = np.concatenate([
        np.stack([1-pred_conf[:,:,1], pred_conf[:,:,1], zero, zero], axis=2),
        np.stack([1-pred_conf[:,:,2], zero, pred_conf[:,:,2], zero], axis=2),
        np.stack([1-pred_conf[:,:,3], zero, zero, pred_conf[:,:,3]], axis=2),
    ], axis=1)
    return x, y, pred_conf

def _process_detections(det_x, det_y, det_p, det_f, wcs, was, wps, eval_r):
    allgts = [wc+wa+wp for wc, wa, wp in zip(wcs, was, wps)]
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(allgts, radius=eval_r)
    wd_r, wd_p, wd_t = prec_rec_2d(np.sum(det_p[:,1:], axis=1), np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(wcs, radius=eval_r)
    # TODO possibly speed up the below significantly since a lot of them have 0 probability by design in some cases and can be dropped.
    wc_r, wc_p, wc_t = prec_rec_2d(det_p[:,1], np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(was, radius=eval_r)
    wa_r, wa_p, wa_t = prec_rec_2d(det_p[:,2], np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(wps, radius=eval_r)
    wp_r, wp_p, wp_t = prec_rec_2d(det_p[:,3], np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)

    return [wd_r, wd_p, wd_t], [wc_r, wc_p, wc_t], [wa_r, wa_p, wa_t], [wp_r, wp_p, wp_t]

def _process_detections_2class(det_x, det_y, det_p, det_f, wcs, was, eval_r):
    allgts = [wc+wa for wc, wa in zip(wcs, was)]
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(allgts, radius=eval_r)
    wd_r, wd_p, wd_t = prec_rec_2d(np.sum(det_p[:,1:], axis=1), np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(wcs, radius=eval_r)
    wc_r, wc_p, wc_t = prec_rec_2d(det_p[:,1], np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)
    gts_x, gts_y, gts_r, gts_f = deep2flat_gt(was, radius=eval_r)
    wa_r, wa_p, wa_t = prec_rec_2d(det_p[:,2], np.c_[det_x, det_y], det_f, np.c_[gts_x, gts_y], gts_f, gts_r)

    return [wd_r, wd_p, wd_t], [wc_r, wc_p, wc_t], [wa_r, wa_p, wa_t]


def comp_prec_rec_softmax(scans, wcs, was, wps, pred_conf, pred_offs, eval_r=0.5, **v2d_kw):
    x, y = _prepare_prec_rec_softmax(scans, pred_offs)
    det_x, det_y, det_p, det_f = deep2flat(votes_to_detections2(x, y, pred_conf, **v2d_kw))

    return _process_detections(det_x, det_y, det_p, det_f, wcs, was, wps, eval_r)



def comp_prec_rec_softmax2(scans, wcs, was, wps, pred_conf, pred_offs, eval_r=0.5, **v2d_kw):
    x, y = _prepare_prec_rec_softmax(scans, pred_offs)
    det_x, det_y, det_p, det_f = deep2flat(votes_to_detections3(x, y, pred_conf, **v2d_kw))

    return _process_detections(det_x, det_y, det_p, det_f, wcs, was, wps, eval_r)


def comp_prec_rec_sigmoids(scans, wcs, was, wps, pred_conf, pred_offs, eval_r=0.5, **v2d_kw):
    x, y, pred_conf = _prepare_prec_rec_sigmoids(scans, pred_offs, pred_conf)
    det_x, det_y, det_p, det_f = deep2flat(votes_to_detections2(x, y, pred_conf, **v2d_kw))

    return _process_detections(det_x, det_y, det_p, det_f, wcs, was, wps, eval_r)


def comp_prec_rec_sigmoids2(scans, wcs, was, wps, pred_conf, pred_offs, eval_r=0.5, **v2d_kw):
    x, y, pred_conf = _prepare_prec_rec_sigmoids(scans, pred_offs, pred_conf)
    det_x, det_y, det_p, det_f = deep2flat(votes_to_detections3(x, y, pred_conf, **v2d_kw))

    return _process_detections(det_x, det_y, det_p, det_f, wcs, was, wps, eval_r)


def peakf1(recs, precs):
    return np.max(2*precs*recs/np.clip(precs+recs, 1e-16, 2+1e-16))


def eer(recs, precs):
    # Find the first nonzero or else (0,0) will be the EER :)
    def first_nonzero_idx(arr):
        return np.where(arr != 0)[0][0]

    p1 = first_nonzero_idx(precs)
    r1 = first_nonzero_idx(recs)
    idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
    return (precs[p1+idx] + recs[r1+idx])/2  # They are often the exact same, but if not, use average.


def plot_prec_rec(wds, wcs, was, wps, figsize=(15,10), title=None):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(*wds[:2], label='agn (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wds[:2], reorder=True), peakf1(*wds[:2]), eer(*wds[:2])), c='#E24A33')
    ax.plot(*wcs[:2], label='wcs (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wcs[:2], reorder=True), peakf1(*wcs[:2]), eer(*wcs[:2])), c='#348ABD')
    ax.plot(*was[:2], label='was (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*was[:2], reorder=True), peakf1(*was[:2]), eer(*was[:2])), c='#988ED5')
    ax.plot(*wps[:2], label='wps (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wps[:2], reorder=True), peakf1(*wps[:2]), eer(*wps[:2])), c='#8EBA42')

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.91)

    prettify_pr_curve(ax)
    lbplt.fatlegend(ax, loc='upper right')
    return fig, ax

# results = comp_prec_rec(va, pred_yva_conf, pred_yva_offs, blur_win=5, blur_sigma=1)
# fig, ax = plot_prec_rec(*results, title='WNet3x 50ep Adam+decay')


def votes_to_detections3(xs, ys, probas, min_thresh=1e-5,
                         bin_size=0.1, blur_win=21, blur_sigma=2.0,
                         x_min=-15.0, x_max=15.0, y_min=-5.0, y_max=15.0,
                         nms_radius=0.2, vote_collect_radius=0.3,
                         weighted_avg=False, retgrid=False,
                         class_weights=None):
    '''
    Convert a list of votes to a list of detections based on Non-Max suppression.
    This version uses a separate voting grid for each class, thus needing an
    additional nms step at the end.

    - `bin_size` the bin size (in meters) used for the grid where votes are cast.
    - `blur_win` the window size (in bins) used to blur the voting grid.
    - `blur_sigma` the sigma used to compute the Gaussian in the blur window.
    - `x_min` the left limit for the voting grid, in meters.
    - `x_max` the right limit for the voting grid, in meters.
    - `y_min` the bottom limit for the voting grid in meters.
    - `y_max` the top limit for the voting grid in meters.
    - `nms_radius` the radius used to suppress less confident maxima.
    - `vote_collect_radius` the radius use during the collection of votes assigned
      to each detection.

    Returns a list of tuples (x,y,probs) where `probs` has the same layout as
    `probas`.
    '''
    if class_weights is not None:
        probas = np.array(probas)  # Make a copy.
        probas[:,:,1:] *= class_weights
    x_range = int((x_max-x_min)/bin_size)
    y_range = int((y_max-y_min)/bin_size)
    grid = np.zeros((x_range, y_range, probas.shape[2]-1), np.float32)

    # Fix the blur_win and blur_sigma if they are scalars

    if isinstance(blur_win, (int, float)) or blur_win is None:
        blur_win = [blur_win] * (probas.shape[2]-1)
    blur_win  = np.asarray(blur_win)
    if len(blur_win) != (probas.shape[2] - 1):
        raise ValueError('Blur window size has to be a scalar or an array with the '
                         'length corresponding to the class count')
    if isinstance(blur_sigma, (int, float)) or blur_sigma is None:
        blur_sigma = [blur_sigma] * (probas.shape[2]-1)
    blur_sigma  = np.asarray(blur_sigma)
    if len(blur_sigma) != (probas.shape[2] - 1):
        raise ValueError('Blur sigma has to be a scalar or an array with the '
                         'length corresponding to the class count')

    vote_collect_radius_sq = vote_collect_radius * vote_collect_radius

    # Update x/y max to correspond to the end of the last bin.
    x_max = x_min + x_range*bin_size
    y_max = y_min + y_range*bin_size

    # Where we collect the outputs.
    all_dets = []
    all_grids = []

    # Iterate over the scans.
    for iscan, (x, y, probs) in enumerate(zip(xs, ys, probas)):
        # Clear the grid, for each scan its own.
        grid.fill(0)
        dets_current = []

        # Filter out all the super-weak votes, as they wouldn't contribute much anyways
        # but waste time.
        voters_idxs = np.where(np.sum(probs[:,1:], axis=-1) > min_thresh)[0]

        # No voters, early bail
        if not len(voters_idxs):
            if retgrid:
                all_grids.append(np.array(grid))  # Be sure to make a copy.
            continue

        x = x[voters_idxs]
        y = y[voters_idxs]
        probs = probs[voters_idxs]

        # Convert x/y to grid-cells.
        x_idx = np.int64((x-x_min)/bin_size)
        y_idx = np.int64((y-y_min)/bin_size)

        mask = (0 <= x_idx) & (x_idx < x_range) & (0 <= y_idx) & (y_idx < y_range)
        x_idx = x_idx[mask]
        x = x[mask]
        y_idx = y_idx[mask]
        y = y[mask]
        probs = probs[mask]

        # Vote into the grid, including the agnostic vote as sum of class-votes!
        np.add.at(grid, [x_idx, y_idx], probs[:,1:])

        # Loop over all classes:
        for c in range(probas.shape[2]-1):
            grid_c = grid[..., c]
            if blur_win[c] is not None and blur_sigma is not None:
                grid_c = cv2.GaussianBlur(grid_c, (blur_win[c],blur_win[c]), blur_sigma[c])
            max_grid = scipy.ndimage.maximum_filter(grid_c, size=3)
            maxima = (grid_c == max_grid) & (grid_c > 0)
            m_x, m_y = np.where(maxima)

            if len(m_x) == 0:
                continue

            # Back from grid-bins to real-world locations.
            m_x = m_x*bin_size + x_min + bin_size/2
            m_y = m_y*bin_size + y_min + bin_size/2

            # For each vote, get which maximum/detection it contributed to.
            # Shape of `center_dist` (ndets, voters) and outer is (voters)
            center_dist = np.square(x - m_x[:,None]) + np.square(y - m_y[:,None])
            det_voters = np.argmin(center_dist, axis=0)

            # Generate the final detections by average over their voters.
            for ipeak in range(len(m_x)):
                # Compute the vote indices, take the closest, but only within a radius.
                my_voter_idxs = np.where(det_voters == ipeak)[0]
                my_voter_idxs = my_voter_idxs[center_dist[ipeak, my_voter_idxs] < vote_collect_radius_sq]

                # Compute the final output for x, y, and probs.
                p = probs[my_voter_idxs, c + 1]
                if weighted_avg:
                    norm = 1 / np.sum(p)
                    new_x = np.sum(x[my_voter_idxs] * p) * norm
                    new_y = np.sum(y[my_voter_idxs] * p) * norm
                    p = np.sum(p * p) * norm
                else:
                    new_x = np.mean(x[my_voter_idxs])
                    new_y = np.mean(y[my_voter_idxs])
                    p = np.mean(p)
                p_padded = np.zeros_like(probs[0])
                p_padded[c + 1] = p
                dets_current.append((new_x, new_y, p_padded))

        # Perform nms on the resulting detections
        keep = np.full([len(dets_current)], True, dtype=np.bool)
        if nms_radius > 0 and len(dets_current) > 0:
            # Store them in a slightly easier format
            all_det_xyp = np.stack([[d[0], d[1], np.max(d[2])] for d in dets_current])

            # Compute the distances between all of them
            dist = cdist(all_det_xyp[:,:2], all_det_xyp[:,:2])

            # Set those that don't influence each other to -1
            dist[dist > nms_radius] = -1
            dist -= np.eye(len(dist))

            # Sort them from strongest to weakest detections.
            det_indices = np.argsort(-all_det_xyp[:,2])
            for d in det_indices:
                if not keep[d]:
                    continue
                # suppress other detections with lower probability
                neighbor_mask = dist[d] > -1
                suppresable_mask = all_det_xyp[:, 2] <= all_det_xyp[d, 2]
                discard = np.logical_and(np.logical_and(neighbor_mask, suppresable_mask), keep)
                for i in np.where(discard)[0]:
                    keep[i] = False
                    dist[i, :] = 0
                    dist[:, i] = 0

        # Store those which passed through the nms.
        all_dets.append([d for d, k in zip(dets_current, keep) if k])

        if retgrid:
            all_grids.append(np.array(grid))  # Be sure to make a copy.

    if retgrid:
        return all_dets, all_grids
    return all_dets


def subsample_pr(precision, recall, dist_threshold):
    p_sample = [precision[0]]
    r_sample = [recall[0]]
    for p, r in zip(precision[1:], recall[1:]):
        if (np.square(p_sample[-1] - p) + np.square(r_sample[-1] - r)) > dist_threshold * dist_threshold:
            p_sample.append(p)
            r_sample.append(r)
    return np.asarray(p_sample), np.asarray(r_sample)


def dump_paper_pr_curves(dump_file, precision, recall,
                         store_prec=3,
                         fast=0.01, fast_postfix='_fast.csv',
                         slow=0.001, slow_postfix='.csv',
                         meta_postfix='_meta.csv'):
    header = 'prec,rec'
    pr_fast = np.asarray(subsample_pr(precision, recall, dist_threshold=fast)).T * 100
    pr_slow = np.asarray(subsample_pr(precision, recall, dist_threshold=slow)).T * 100
    a = auc(recall, precision, reorder=True) * 100
    f1 = peakf1(recall, precision) * 100
    e = eer(recall, precision) * 100
    meta = np.asarray([a, f1, e])[None]

    np.savetxt(dump_file + fast_postfix, pr_fast, fmt='%.{}f'.format(store_prec), delimiter=',', header=header, comments='')
    np.savetxt(dump_file + slow_postfix, pr_slow, fmt='%.{}f'.format(store_prec), delimiter=',', header=header, comments='')
    np.savetxt(dump_file + meta_postfix, meta, fmt='%.{}f'.format(store_prec), delimiter=',', header='auc,f1,eer', comments='')


import signal
import multiprocessing

class BackgroundFunction:
    def __init__(self, function, prefetch_count, reseed=True, **kwargs):
        """Parallelize a function to prefetch results using mutliple processes.
        Args:
            function: Function to be executed in parallel.
            prefetch_count: Number of samples to prefetch.
            kwargs: Keyword args passed to the executed function.

        NOTE: This is taken from Alexander Hermans at
              https://github.com/Pandoro/tools/blob/master/utils.py
        """
        self.function = function
        self.prefetch_count = prefetch_count
        self.kwargs = kwargs
        self.output_queue = multiprocessing.Queue(maxsize=prefetch_count)
        self.procs = []
        for i in range(self.prefetch_count):
            p = multiprocessing.Process(
                target=BackgroundFunction._compute_next,
                args=(self.function, self.kwargs, self.output_queue, reseed))
            p.daemon = True  # To ensure it is killed if the parent dies.
            p.start()
            self.procs.append(p)

    def fill_status(self, normalize=False):
        """Returns the fill status of the underlying queue.
        Args:
            normalize: If set to True, normalize the fill status by the max
                queue size. Defaults to False.
        Returns:
            The possibly normalized fill status of the underlying queue.
        """
        return (self.output_queue.qsize() /
            (self.prefetch_count if normalize else 1))

    def __call__(self):
        """Obtain one of the prefetched results or wait for one.
        Returns:
            The output of the provided function and the given keyword args.
        """
        output = self.output_queue.get(block=True)
        return output

    def __del__(self):
        """Signal the processes to stop and join them."""
        for p in self.procs:
            p.terminate()
            p.join()

    def _compute_next(function, kwargs, output_queue, reseed):
        """Helper function to do the actual computation in a non_blockig way.
        Since this will always run in a new process, we ignore the interrupt
        signal for the processes. This should be handled by the parent process
        which kills the children when the object is deleted.
        Some more discussion can be found here:
        https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if reseed:
            np.random.seed()

        while True:
            output_queue.put(function(**kwargs))


####
# FOR FINAL EXPS
###

def generate_votes(scan, wcs, was, wps, rwc=0.6, rwa=0.4, rwp=0.35, lblwc=1, lblwa=2, lblwp=3):
    N = len(scan)
    y_conf = np.zeros( N, dtype=np.int64)
    y_offs = np.zeros((N, 2), dtype=np.float32)

    alldets = list(wcs) + list(was) + list(wps)
    radii = [rwc]*len(wcs) + [rwa]*len(was) + [rwp]*len(wps)
    dets = closest_detection(scan, alldets, radii)
    labels = [0] + [lblwc]*len(wcs) + [lblwa]*len(was) + [lblwp]*len(wps)

    for i, (r, phi) in enumerate(zip(scan, laser_angles(N))):
        if 0 < dets[i]:
            y_conf[i] = labels[dets[i]]
            y_offs[i,:] = global2win(r, phi, *alldets[dets[i]-1])

    return y_conf, y_offs


from functools import partial


class Dataset:
    def __init__(self, filenames, DATADIR, LABELDIR, **votegenkw):
        self.scansns, self.scants, self.scans = zip(*[load_scan(f + '.csv') for f in filenames])
        self.detsns, self.wcdets, self.wadets, self.wpdets = zip(*map(
            lambda f: load_dets(f, DATADIR, LABELDIR), filenames))
        self.odoms = [load_odom(f + '.odom2') for f in filenames]

        # Pre-compute mappings from detection index to scan index.
        self.idet2iscan = [{i: np.where(ss == d)[0][0] for i, d in enumerate(ds)}
                               for ss, ds in zip(self.scansns, self.detsns)]

        # This is in order to pick uniformly across annotated scans further down.
        # It's significantly faster this way than computing sequence-probabilities and sampling that way.
        self._seq_picker = np.concatenate([[i]*len(sns) for i, sns in enumerate(self.detsns)])

        # Targets. Kinda ugly, but at least correct, unlike what I had before: pretty but wrong!
        self.y_conf, self.y_offs = [], []
        for iseq, detsns in enumerate(self.detsns):
            y_confs, y_offss = [], []
            for idet, detsn in enumerate(detsns):
                y_conf, y_offs = generate_votes(
                    self.scans[iseq][self.idet2iscan[iseq][idet]],
                    self.wcdets[iseq][idet], self.wadets[iseq][idet], self.wpdets[iseq][idet],
                    **votegenkw)
                y_confs.append(y_conf)
                y_offss.append(y_offs)
            self.y_conf.append(np.array(y_confs))
            self.y_offs.append(np.array(y_offss))

    def random_index(self, min_before=0):
        iseq = np.random.choice(len(self._probs), p=self._probs)
        iscan = min_before + np.random.choice(len(self.scans[iseq]) - min_before)
        return iseq, iscan

    def random_labelled_index(self, min_before=0):
        iseq = np.random.choice(self._seq_picker)
        detsns = self.detsns[iseq]

        # Figure out for how many labelled scans we don't have enough scans before.
        # Do so using the sequence-number, and we know they are sorted.
        scan0 = self.scansns[iseq][0]
        for skip, sn in enumerate(detsns):
            if scan0 <= sn:
                break
        idet = np.random.randint(skip, len(detsns))
        return iseq, idet, self.idet2iscan[iseq][idet]


def cutout(scans, odoms, ipoint, out=None, odom='rot-rel', win_sz=1.66, thresh_dist=1,
           center='point', center_time='now', value='donut', nsamp=48, UNK=29.99, laserIncrement=laserIncrement):
    """ TODO: Probably we can still try to clean this up more.
    This function here only creates a single cut-out; for training,
    we'd want to get a batch of cutouts from each seq (can vectorize) and for testing
    we'd want all cutouts for one scan, which we can vectorize too.
    But ain't got time for this shit!

    Args:
    - scans: (T,N) the T scans (of scansize N) to cut out from, `T=-1` being the "current time".
    - out: None or a (T,nsamp) buffer where to store the cutouts.
    """
    T, N = scans.shape

    # Compute the size (width) of the window at the last time index:
    z = scans[-1,ipoint]
    half_alpha = float(np.arctan(0.5*win_sz/z))

    # Pre-allocate some buffers
    out = np.zeros((T,nsamp), np.float32) if out is None else out
    SCANBUF = np.full(N+1, UNK, np.float32)  # Pad by UNK for the border-padding by UNK.
    for t in range(T):
        # If necessary, compute the odometry of the current time relative to the "key" one.
        # TODO: in principle we could also interpolate using the time, since they don't
        #       *exactly* line up with the scan's times.
        if odom is not False:
            odom_x, odom_y, odom_a = map(float, odoms[t]['xya'] - odoms[-1]['xya'])
        else:
            odom_x, odom_y, odom_a = 0.0, 0.0, 0.0

        # Compute the start and end indices of points in the scan to be considered.
        start = int(round(ipoint - half_alpha/laserIncrement - odom_a/laserIncrement))
        end = int(round(ipoint + half_alpha/laserIncrement - odom_a/laserIncrement))

        # Now compute the list of indices at which to take the points,
        # using -1/end to access out-of-bounds which has been set to UNK.
        support_points = np.arange(start, end+1)
        support_points.clip(-1, len(SCANBUF)-1, out=support_points)

        # Write the scan into the buffer which has UNK at the end and then sample from it.
        SCANBUF[:-1] = scans[t]
        cutout = SCANBUF[support_points]

        # Now in case we want to apply "translation" odometry, the best effort we can do,
        # is to project the relative odometry onto the "front" vector of the cutout, and
        # apply that onto the radius (`z`) of the points.
        # TODO: Maybe we can actually do better in the 'undistorted' case below.
        if odom in ('full', 'full-rel'):
            #cutout += np.dot([np.cos(odom_a), np.sin(odom_a)], [odom_x, odom_y])
            cutout += np.cos(odom_a)*odom_x + np.sin(odom_a)*odom_y

        # Now we do the resampling of the cutout to a fixed number of points. We can do it two ways:
        if 'undistorted' in value:
            # In the 'undistorted' case, we actually use x/y cartesian space, i.e. the cut-out
            # is not arc-shaped but really rectangle-shaped.
            # For doing this, we need real interpolation-functionality since even the 'x' axis
            # will be converted non-linearly from angles to points on a line.
            dcorr_a = np.linspace(-half_alpha, half_alpha, len(cutout))# - odom_a
            y = np.cos(dcorr_a) * cutout
            x = np.sin(dcorr_a) * cutout
            kw = {'fill_value': 'extrapolate'} if '(extra)' in value else {'bounds_error': False, 'fill_value': UNK}
            interp = scipy.interpolate.interp1d(x, y, assume_sorted=False, copy=False, kind='linear', **kw)
            out[t] = interp(np.linspace(-0.5*win_sz, 0.5*win_sz, nsamp))
        else:
            # In the other case, we have a somewhat distorted world-view as the x-indices
            # correspond to angles and the values to z-distances (radii) as in original DROW.
            # The advantage here is we can use the much faster OpenCV resizing functions.
            interp = cv2.INTER_AREA if nsamp < len(cutout) else cv2.INTER_LINEAR
            cv2.resize(cutout[None], (nsamp,1), interpolation=interp, dst=out[None,t])

        # Now we choose where to center the depth at before we will re-center/clip.
        if center_time == 'each':
            z = scans[t][ipoint]

        # Clip things too close and too far to create the "focus tunnel" since they are likely irrelevant.
        out[t].clip(z - thresh_dist, z + thresh_dist, out=out[t])
        #fastclip_(cutouts[i], z - thresh_dist, z + thresh_dist)

        # And finally, possibly re-align according to a few different choices.
        if center == 'point':
            out[t] -= z
        elif center == 'near':
            out[t] -= z - thresh_dist
        elif center == 'far':
            out[t] = (z + thresh_dist) - out[t]

    return out


def get_batch(data, bs, ntime, nsamp, dtime=1, repeat_before=True, **cutout_kw):
    Xb = np.empty((bs, ntime, nsamp), np.float32)
    yb_conf = np.empty(bs, np.int64)
    yb_offs = np.empty((bs, 2), np.float32)

    for b in range(bs):
        if repeat_before:
            # Prepend the exact same scan/odom for the first few where there's no history.
            iseq, idet, iscan = data.random_labelled_index()
            times = np.arange(iscan - ntime*dtime + 1, iscan+1, dtime)
            times[times < 0] = times[0 <= times][0]

            scans = np.array([data.scans[iseq][j] for j in times])
            odoms = np.array([data.odoms[iseq][j] for j in times])
        else:
            iseq, idet, iscan = data.random_labelled_index(min_before=(ntime-1)*dtime)
            scans = data.scans[iseq][iscan-(ntime-1)*dtime:iscan+1:dtime]
            odoms = data.odoms[iseq][iscan-(ntime-1)*dtime:iscan+1:dtime]

        ipt = np.random.randint(len(scans[0]))
        cutout(scans, odoms, ipt, out=Xb[b], nsamp=nsamp, **cutout_kw)

        yb_conf[b] = data.y_conf[iseq][idet][ipt]
        yb_offs[b] = data.y_offs[iseq][idet][ipt]

    return Xb, yb_conf, yb_offs
