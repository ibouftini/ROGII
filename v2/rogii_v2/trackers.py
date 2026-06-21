"""Numba-accelerated trackers: Particle Filter, Beam Search, NCC."""
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    """Linear interpolation on a regular grid. Numba-compatible."""
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    """Systematic resampling with jitter."""
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


def make_grid(tw_tvt, tw_gr, step=0.2):
    """Create regular grid for fast Numba interpolation."""
    vmin = float(tw_tvt.min())
    vmax = float(tw_tvt.max())
    n = int((vmax - vmin) / step) + 2
    grid_tvt = np.linspace(vmin, vmin + (n - 1) * step, n)
    grid_gr = np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float64)
    return grid_gr, vmin, step


# ---------------------------------------------------------------------------
# Particle Filter -- ANCC-anchored
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP):
    """ANCC particle filter. Tracks in U-space (pos = TVT + Z).
    Returns (tvt_trajectory, std_trajectory)."""
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        if dm < 1.0:
            dm = 1.0
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            if tvt_j < vmin - 50.0:
                tvt_j = vmin - 50.0
            if tvt_j > vmin + len(gg) * step + 50.0:
                tvt_j = vmin + len(gg) * step + 50.0
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                if d * d < 600.0:
                    lk = np.exp(-0.5 * d * d)
                else:
                    lk = 0.0
                if lk < 1e-300:
                    lk = 1e-300
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_


# ---------------------------------------------------------------------------
# Particle Filter -- Z-velocity-coupled
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs,
          ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    """Z-velocity PF. Uses dual GR channels and Z-velocity model."""
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        if dm < 1.0:
            dm = 1.0
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            if pos[j] < vmin - 50.0:
                pos[j] = vmin - 50.0
            if pos[j] > vmin + len(gg_p) * step + 50.0:
                pos[j] = vmin + len(gg_p) * step + 50.0
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                if dp * dp < 600.0:
                    lp = np.exp(-0.5 * dp * dp)
                else:
                    lp = 0.0
                if lp < 1e-300:
                    lp = 1e-300
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    if ds * ds < 600.0:
                        lsm = np.exp(-0.5 * ds * ds)
                    else:
                        lsm = 0.0
                    if lsm < 1e-300:
                        lsm = 1e-300
                    lk = (1.0 - GR_WT) * lp + GR_WT * lsm
                else:
                    lk = lp
                if lk < 1e-300:
                    lk = 1e-300
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        # Z-velocity penalty
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            if dv * dv < 600.0:
                lz = np.exp(-0.5 * dv * dv)
            else:
                lz = 0.0
            if lz < 1e-300:
                lz = 1e-300
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


# ---------------------------------------------------------------------------
# 128-seed Likelihood-weighted PF Ensemble
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
                     n_seeds, seed_base, MOM, VN, PN, RP, RR, RESAMP, init_spr):
    """Run n_seeds independent PF seeds. Returns (preds[n_seeds, T], liks[n_seeds])."""
    n = len(md_v)
    preds = np.empty((n_seeds, n))
    liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N)
        rate = np.empty(N)
        w = np.ones(N) / N
        for j in range(N):
            pos[j] = ls + init_spr * np.random.randn()
            rate[j] = ir + 0.01 * np.random.randn()
        log_lik = 0.0
        prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0
            for j in range(N):
                rate[j] = MOM * rate[j] + VN * np.random.randn()
                pos[j] += rate[j] * dm + PN * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.0:
                    tvt_j = vmin - 100.0
                if tvt_j > tmax + 100.0:
                    tvt_j = tmax + 100.0
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                dd = d * d
                if dd > 600.0:
                    dd = 600.0
                lk = np.exp(-0.5 * dd)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += w[j] * lk
                w[j] = w[j] * lk
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N):
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
            neff = 0.0
            for j in range(N):
                neff += w[j] * w[j]
            neff = 1.0 / neff
            if neff < RESAMP * N:
                cum = np.empty(N)
                c = 0.0
                for j in range(N):
                    c += w[j]
                    cum[j] = c
                u0 = np.random.uniform(0.0, 1.0 / N)
                newpos = np.empty(N)
                newrate = np.empty(N)
                ci = 0
                for j in range(N):
                    u = u0 + j / N
                    while ci < N - 1 and cum[ci] < u:
                        ci += 1
                    newpos[j] = pos[ci] + RP * np.random.randn()
                    newrate[j] = rate[ci] + RR * np.random.randn()
                for j in range(N):
                    pos[j] = newpos[j]
                    rate[j] = newrate[j]
                    w[j] = 1.0 / N
            est = 0.0
            for j in range(N):
                est += w[j] * (pos[j] - z_v[i])
            preds[s, i] = est
            prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search on typewell grid. Returns path of typewell indices."""
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for t in range(n):
        gv = sgr[t]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[t, :kept] = cI[:kept]
        hP[t, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def run_pf_ancc(hw, tw_tvt, tw_gr, cfg):
    """Run single PF ANCC. Returns (tvt_eval, std_eval)."""
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10:
        return np.array([]), np.array([])
    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0))
    tail = kn.tail(30).dropna(subset=['TVT_input'])
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, vmin, step = make_grid(tw_tvt, tw_gr)
    gr_interp = hw.GR.interpolate(limit_direction='both').fillna(float(tw_gr.mean()))
    gr_v = gr_interp.values[ev.index].astype(np.float64)
    md_v = ev.MD.values.astype(np.float64)
    z_v = ev.Z.values.astype(np.float64)
    tvt, std = _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir,
                        cfg.PF_N_PARTICLES,
                        cfg.PF_ANCC_ALPHA, cfg.PF_ANCC_RN, cfg.PF_ANCC_PN,
                        cfg.PF_ANCC_IS, cfg.PF_ANCC_RP, cfg.PF_ANCC_RR,
                        cfg.PF_ANCC_RESAMP)
    return tvt.astype(np.float32), std.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, cfg):
    """Run single PF Z-velocity. Returns (tvt_eval, std_eval)."""
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10:
        return np.array([]), np.array([])
    last_tvt = float(kn.iloc[-1].TVT_input)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0))
    tail = kn.tail(30).dropna(subset=['TVT_input'])
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    m = dm > 0
    iv = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    # Fit Z-velocity model from known zone
    kn_clean = kn.dropna(subset=['TVT_input'])
    kmd = kn_clean.MD.values
    kz = kn_clean.Z.values
    ktvt = kn_clean.TVT_input.values
    dmd_k = np.diff(kmd)
    dz_k = np.diff(kz)
    dtvt_k = np.diff(ktvt)
    valid = dmd_k > 0
    if valid.sum() >= 5:
        dzd_k = dz_k[valid] / dmd_k[valid]
        dtvt_dmd_k = dtvt_k[valid] / dmd_k[valid]
        A = np.column_stack([dzd_k, np.ones(len(dzd_k))])
        coeffs, _, _, _ = np.linalg.lstsq(A, dtvt_dmd_k, rcond=None)
        beta, icpt = float(coeffs[0]), float(coeffs[1])
        resid = dtvt_dmd_k - (beta * dzd_k + icpt)
        zsig = float(np.std(resid))
    else:
        beta, icpt, zsig = 0.0, 0.0, 0.1
    gg, vmin, step = make_grid(tw_tvt, tw_gr)
    gr_interp = hw.GR.interpolate(limit_direction='both').fillna(float(tw_gr.mean()))
    gr_v = gr_interp.values[ev.index].astype(np.float64)
    # Smoothed GR for dual-channel
    gr_sm = pd.Series(gr_interp.values).rolling(int(cfg.PF_Z_GR_WT * 10 + 1),
            center=True, min_periods=1).mean().values[ev.index].astype(np.float64)
    # Smoothed typewell grid
    tw_gr_sm = pd.Series(tw_gr).rolling(5, center=True, min_periods=1).mean().values
    gg_s, _, _ = make_grid(tw_tvt, tw_gr_sm.astype(np.float64), step)
    md_v = ev.MD.values.astype(np.float64)
    z_v = ev.Z.values.astype(np.float64)
    tvt, std = _pf_z(md_v, z_v, gr_v, gr_sm, gg, gg_s, vmin, step, gs,
                     last_tvt, iv, beta, icpt, zsig,
                     cfg.PF_N_PARTICLES,
                     cfg.PF_Z_MOM, cfg.PF_Z_VN, cfg.PF_Z_PN,
                     cfg.PF_Z_GR_WT, cfg.PF_Z_RP, cfg.PF_Z_RV, cfg.PF_Z_RESAMP)
    return tvt.astype(np.float32), std.astype(np.float32)


def run_lik_pf(hw, tw_tvt, tw_gr, cfg, seed_base=0):
    """128-seed likelihood-weighted PF ensemble.
    Returns dict of scale-weighted predictions + mean, and eval index."""
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10:
        return {}, np.array([])
    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0))
    tail = kn.tail(30).dropna(subset=['TVT_input'])
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, vmin, step = make_grid(tw_tvt, tw_gr)
    gr_interp = hw.GR.interpolate(limit_direction='both').fillna(float(tw_gr.mean()))
    gr_v = gr_interp.values[ev.index].astype(np.float64)
    md_v = ev.MD.values.astype(np.float64)
    z_v = ev.Z.values.astype(np.float64)
    preds, liks = _pf_lik_allseeds(
        md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir,
        cfg.PF_N_PARTICLES, cfg.PF_N_SEEDS, seed_base,
        cfg.PF_ANCC_ALPHA, cfg.PF_ANCC_RN, cfg.PF_ANCC_PN,
        cfg.PF_ANCC_RP, cfg.PF_ANCC_RR, cfg.PF_ANCC_RESAMP,
        cfg.PF_INIT_SPREAD,
    )
    ln = liks - liks.max()
    out = {}
    for sc in cfg.PF_SCALES:
        wts = np.exp(ln / float(sc))
        wts /= wts.sum()
        out[f'scale_{sc:g}'] = (wts[:, None] * preds).sum(0).astype(np.float32)
    out['mean'] = preds.mean(0).astype(np.float32)
    return out, ev.index.values


def beam_search(hw_gr, tw_tvt, tw_gr, last_tvt, bs, mc, es, smooth_r):
    """Single beam search config. Returns TVT trajectory for eval zone."""
    if smooth_r > 0:
        k = np.ones(smooth_r * 2 + 1) / (smooth_r * 2 + 1)
        sgr = np.convolve(hw_gr, k, 'same').astype(np.float64)
    else:
        sgr = hw_gr.astype(np.float64)
    si = int(np.searchsorted(tw_tvt, last_tvt))
    si = max(0, min(si, len(tw_tvt) - 1))
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, mc, es)
    return tw_tvt[path].astype(np.float32)


def run_beam_configs(hw_gr, tw_tvt, tw_gr, last_tvt, configs):
    """Run all beam configs. Returns dict of name -> TVT trajectory."""
    out = {}
    for bs, mc, es, sr, name in configs:
        out[name] = beam_search(hw_gr, tw_tvt, tw_gr, last_tvt, bs, mc, es, sr)
    out['beam_ref'] = (out['cons'] + out['sm5']) / 2.0
    return out


def multi_scale_ncc(kgr, ktvt, hgr, tw_tvt=None, tw_gr=None, hws=(8, 15, 25), stride=3):
    """Multi-scale NCC. Returns list of (tvt, scores) per scale + ensemble."""
    results = []
    for hw_size in hws:
        n = len(hgr)
        tvt_out = np.full(n, np.nan, dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(0, n, stride):
            lo = max(0, i - hw_size)
            hi = min(n, i + hw_size + 1)
            hw_win = hgr[lo:hi]
            if len(hw_win) < 3:
                continue
            best_score = -1.0
            best_tvt = ktvt[-1] if len(ktvt) else 0.0
            for j in range(len(kgr) - len(hw_win) + 1):
                kw = kgr[j:j + len(hw_win)]
                hw_c = hw_win - hw_win.mean()
                kw_c = kw - kw.mean()
                denom = np.sqrt((hw_c ** 2).sum() * (kw_c ** 2).sum())
                if denom < 1e-8:
                    continue
                ncc = float((hw_c * kw_c).sum() / denom)
                if ncc > best_score:
                    best_score = ncc
                    mid_j = j + len(hw_win) // 2
                    best_tvt = ktvt[min(mid_j, len(ktvt) - 1)]
            tvt_out[i] = best_tvt
            scores[i] = max(best_score, 0.0)
        # Interpolate gaps from stride
        valid = ~np.isnan(tvt_out)
        if valid.sum() > 1:
            idx = np.arange(n)
            tvt_out = np.interp(idx, idx[valid], tvt_out[valid]).astype(np.float32)
        elif valid.sum() == 1:
            tvt_out[:] = tvt_out[valid][0]
        else:
            tvt_out[:] = ktvt[-1] if len(ktvt) else 0.0
        # Interpolate scores too
        if valid.sum() > 1:
            scores = np.interp(idx, idx[valid], scores[valid]).astype(np.float32)
        results.append((tvt_out, scores))
    # Score-weighted ensemble
    all_tvt = np.stack([r[0] for r in results])
    all_sc = np.stack([r[1] for r in results])
    sw = np.exp(3.0 * all_sc)
    sw /= sw.sum(axis=0, keepdims=True) + 1e-10
    ensemble = (sw * all_tvt).sum(axis=0).astype(np.float32)
    return results, ensemble


# Warm up Numba JIT (call once at import with tiny arrays)
def _warmup():
    m = np.linspace(1, 50, 20)
    z = np.zeros(20)
    g = np.full(20, 50.0)
    gg = np.linspace(45, 55, 100)
    _pf_ancc(m, z, g, gg, 45.0, 0.1, 20.0, 50.0, 0.0, 8,
             0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
    _beam_jit(g, gg[:20], 10, 5, 10.0, 100.0)
    _pf_lik_allseeds(m, z, g, gg, 45.0, 0.1, 20.0, 50.0, 0.0, 8,
                     2, 0, 0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5)

import pandas as pd  # needed for run_pf_z smoothing

try:
    _warmup()
except Exception:
    pass  # will compile on first real call
