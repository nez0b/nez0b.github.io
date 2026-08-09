#!/usr/bin/env python3
"""Regenerate the neutral-atom series figures from sanitized saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.linalg import expm


COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "gray": "#667085", "light": "#EAECF0", "ink": "#172033",
}
METHOD_COLORS = {
    "GRAPE": COLORS["blue"],
    "open-system GRAPE": COLORS["sky"], "robust GRAPE": COLORS["purple"],
    "Krotov": COLORS["orange"],
    "CRAB": COLORS["green"],
    "Levine–Pichler": "#8B5CF6", "analytic Levine–Pichler": "#8B5CF6",
    "collocation 0.9999": COLORS["red"], "Jandura–Pupillo": "#344054",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "figure.facecolor": "white",
        "axes.facecolor": "white", "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.titleweight": "semibold",
        "axes.edgecolor": "#98A2B3", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E4E7EC", "grid.linewidth": 0.65,
        "grid.alpha": 0.8, "legend.frameon": False, "lines.linewidth": 1.65,
        "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"],
        "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
        "svg.hashsalt": "neutral-atom-control-v2", "savefig.transparent": False,
    })


def panel_labels(axes) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        ax.text(.015, .975, label, transform=ax.transAxes, weight="bold", fontsize=10,
                ha="left", va="top", clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 1.0})


def bounds_check(fig: mpl.figure.Figure, stem: str) -> None:
    fig.canvas.draw()
    # Constrained layout may settle axis positions on the first draw.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders = []
    artists = list(fig.findobj(mpl.text.Text)) + list(fig.findobj(mpl.legend.Legend))
    artists += list(fig.findobj(FancyArrowPatch)) + list(fig.findobj(FancyBboxPatch))
    tick_text = {label for ax in fig.axes for label in (*ax.get_xticklabels(), *ax.get_yticklabels())}
    for artist in artists:
        if not artist.get_visible() or (isinstance(artist, mpl.text.Text) and not artist.get_text()):
            continue
        if artist in tick_text:
            continue
        try:
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        if box.width == 0 and box.height == 0:
            continue
        # Matplotlib keeps off-view tick Text artists alive; they are clipped and
        # never painted, so only test artists that intersect the visible canvas.
        if box.x1 <= canvas.x0 or box.y1 <= canvas.y0 or box.x0 >= canvas.x1 or box.y0 >= canvas.y1:
            continue
        if box.x0 < canvas.x0 - 2 or box.y0 < canvas.y0 - 2 or box.x1 > canvas.x1 + 2 or box.y1 > canvas.y1 + 2:
            label = artist.get_text() if isinstance(artist, mpl.text.Text) else ""
            offenders.append(f"{type(artist).__name__}:{label[:24]}")
    if offenders:
        raise RuntimeError(f"{stem}: artists outside canvas: {offenders[:8]}")


def save(fig: mpl.figure.Figure, out_dir: Path, stem: str, check: bool = True) -> None:
    if check:
        bounds_check(fig, stem)
    metadata = {"Date": None, "Creator": "neutral-atom-control figure generator"}
    fig.savefig(out_dir / f"{stem}.svg", metadata=metadata, facecolor="white")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, metadata={"Software": "neutral-atom-control figure generator"}, facecolor="white")
    plt.close(fig)


def step_time(record: dict) -> np.ndarray:
    return (np.arange(record["N"]) + 0.5) * record["dt_us"]


def hamiltonian_parts(g1: dict):
    p = np.zeros((3, 3), complex); p[1, 2] = 1
    nr = np.diag([0, 0, 1]).astype(complex); ident = np.eye(3)
    hx = .5 * (np.kron(p + p.T, ident) + np.kron(ident, p + p.T))
    hy = .5 * (np.kron(1j*p - 1j*p.T, ident) + np.kron(ident, 1j*p - 1j*p.T))
    hd = -(np.kron(nr, ident) + np.kron(ident, nr))
    h0 = g1["constants"]["V_canonical"] * np.kron(nr, nr)
    return hx, hy, hd, h0


def part1_baselines(data: dict, out_dir: Path) -> None:
    lp = data["levine_pichler"]; segments = lp["segments"]
    t = np.array([0, segments[0]["duration_us"], lp["duration_us"]])
    phases = np.array([segments[0]["phi"], segments[1]["phi"]])
    omega = segments[0]["omega"] / (2*np.pi); delta = segments[0]["delta"] / (2*np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)
    ax = axes[0]
    ax.stairs(omega*np.cos(phases), t, label=r"$\Omega_x/2\pi$", color=COLORS["blue"])
    ax.stairs(omega*np.sin(phases), t, label=r"$\Omega_y/2\pi$", color=COLORS["orange"])
    ax.stairs([delta, delta], t, label=r"$\Delta/2\pi$", color=COLORS["green"])
    ax.axvline(t[1], color=COLORS["gray"], ls=":", lw=1)
    ax.annotate(r"phase jump $\xi$", xy=(t[1], 1.45), xytext=(.41, 1.72), arrowprops={"arrowstyle":"->", "lw":.8}, fontsize=8)
    ax.set(xlabel=r"time [$\mu$s]", ylabel="control [MHz]", title="Analytic two-segment pulse")
    ax.legend(ncol=2, fontsize=7.6, loc="lower left")
    spacing = np.array([float(x) for x in lp["infidelity_vs_spacing"]]); infid = np.array(list(lp["infidelity_vs_spacing"].values())); order=np.argsort(spacing)
    ax = axes[1]; ax.semilogy(spacing[order], infid[order], "o-", color=COLORS["purple"])
    ax.axvline(data["constants"]["r_canonical_um"], color=COLORS["gray"], ls="--", lw=1)
    ax.annotate("canonical 5 µm", xy=(5, infid[order][0]), xytext=(5.25, .012), arrowprops={"arrowstyle":"->", "lw":.8}, fontsize=8)
    ax.set(xlabel=r"spacing [$\mu$m]", ylabel=r"$1-F_{\mathrm{tr}}$", title="Finite-blockade rollout")
    panel_labels(axes); save(fig, out_dir, "part1-analytic-pulse-blockade")


def part1_blockade_population(data: dict, out_dir: Path) -> None:
    hx, hy, hd, h0 = hamiltonian_parts(data); lp=data["levine_pichler"]
    dt=.001; times=np.arange(0, lp["duration_us"]+dt/2, dt); psi=np.eye(9,dtype=complex)[:,4]
    traces={"11":[],"W":[],"rr":[],"other":[]}
    for time in times:
        traces["11"].append(abs(psi[4])**2); traces["W"].append(abs(psi[5])**2+abs(psi[7])**2); traces["rr"].append(abs(psi[8])**2)
        traces["other"].append(max(0,1-traces["11"][-1]-traces["W"][-1]-traces["rr"][-1]))
        if time < times[-1]:
            seg=lp["segments"][0 if time < lp["tau_us"] else 1]
            x=seg["omega"]*np.cos(seg["phi"]); y=seg["omega"]*np.sin(seg["phi"])
            psi=expm(-1j*(h0+x*hx+y*hy+seg["delta"]*hd)*dt)@psi
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.55),constrained_layout=True)
    ax=axes[0]; ax.grid(False); ax.set_aspect("equal")
    p1=np.array([-.55,0]); p2=np.array([.55,0])
    for p in (p1,p2):
        ax.add_patch(Ellipse(p,1.55,.72,angle=20,fc=COLORS["purple"],ec=COLORS["purple"],alpha=.12,lw=1.4)); ax.scatter(*p,s=230,c=COLORS["sky"],edgecolor="white",lw=1.5,zorder=4)
    ax.plot([p1[0],p2[0]],[0,0],color=COLORS["orange"],lw=2); ax.annotate(r"$V=C_6/R^6$",(0,.07),ha="center",fontsize=9)
    ax.annotate("global drive",xy=(-.38,.10),xytext=(-1.18,.92),arrowprops={"arrowstyle":"->","color":COLORS["blue"]},color=COLORS["blue"],fontsize=8)
    ax.set(xlim=(-1.55,1.55),ylim=(-.75,1.2),title="Overlapping blockade volumes"); ax.axis("off")
    ax=axes[1]
    for key,label,color in (("11",r"$|11\rangle$",COLORS["blue"]),("W",r"single Rydberg",COLORS["green"]),("rr",r"$|rr\rangle$",COLORS["red"]),("other","other",COLORS["gray"])):
        ax.plot(1000*times,traces[key],label=label,color=color)
    ax.axvline(1000*lp["tau_us"],color=COLORS["gray"],ls=":",lw=1)
    ax.set(xlabel="time [ns]",ylabel="population",ylim=(-.03,1.03),title=r"Analytic-pulse evolution from $|11\rangle$"); ax.legend(fontsize=7.5,ncol=2)
    panel_labels(axes); save(fig,out_dir,"part1-blockade-populations")


def part2_pulses(data: dict, out_dir: Path) -> None:
    methods=[("GRAPE",data["grape_cz"]),("Krotov",data["krotov_cz"]),("CRAB",data["crab_cz"])]
    fig,axes=plt.subplots(3,1,figsize=(9.2,6.55),constrained_layout=True)
    handles = None
    for ax,(name,r) in zip(axes,methods):
        time=step_time(r); label=r"$\leq10^{-13}$" if r["infidelity"]==0 else f"{r['infidelity']:.2e}"
        ax.plot(time,np.asarray(r["omega_x"])/(2*np.pi),color=COLORS["blue"],label=r"$\Omega_x/2\pi$ · drive quadrature")
        ax.plot(time,np.asarray(r["omega_y"])/(2*np.pi),color=COLORS["orange"],label=r"$\Omega_y/2\pi$ · drive quadrature")
        ax.plot(time,np.asarray(r["delta"])/(2*np.pi),color=COLORS["green"],label=r"$\Delta/2\pi$ · detuning")
        ax.set(ylabel="control / $2\pi$\n[MHz]",title=f"{name} · {r['T_us']*1000:.0f} ns · 1−Ftr {label}")
        if handles is None:
            handles = ax.get_lines()
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=7.5)
    axes[-1].set_xlabel(r"time [$\mu$s]"); panel_labels(axes)
    save(fig,out_dir,"part2-method-pulses")


def part2_reference(data: dict, jp: dict, rough: dict, out_dir: Path) -> None:
    records=[("GRAPE",data["grape_cz"]),("CRAB",data["crab_cz"]),("Levine–Pichler",data["lp_cz"])]
    fig,axes=plt.subplots(2,1,figsize=(9.0,5.6),constrained_layout=True,sharex=False)
    roughness={r["name"]:r["normalized_quadrature_variation"] for r in rough["pulses"]}
    for name,r in records:
        t=1000*step_time(r); x=np.asarray(r["omega_x"]); y=np.asarray(r["omega_y"])
        key="analytic Levine–Pichler" if name=="Levine–Pichler" else name
        label=f"{name} · variation {roughness[key]:.2f}"
        axes[0].plot(t,np.hypot(x,y)/(2*np.pi),color=METHOD_COLORS[name],label=label,ls="--" if name=="Levine–Pichler" else "-")
        axes[1].plot(t,np.unwrap(np.arctan2(y,x)),color=METHOD_COLORS[name],ls="--" if name=="Levine–Pichler" else "-")
    pub=jp["pulses"]["time_optimal_cz"]; tpub=np.asarray(pub["t_omega_max"])/ (2*np.pi*2)*1000
    amp=np.asarray(pub["amplitude_over_omega_max"])*2; phase=np.unwrap(np.asarray(pub["phase_rad"]))
    label=f"Jandura–Pupillo · variation {roughness['Jandura–Pupillo 01_cz']:.2f}"
    axes[0].plot(tpub,amp,color=METHOD_COLORS["Jandura–Pupillo"],label=label)
    axes[1].plot(tpub,phase,color=METHOD_COLORS["Jandura–Pupillo"])
    axes[0].set(ylabel=r"$|\Omega|/2\pi$ [MHz]",title="Saved and published pulse amplitudes"); axes[0].legend(fontsize=7.1,ncol=2)
    axes[1].set(xlabel="time [ns]",ylabel="unwrapped phase [rad]",title="Phase exposes structure hidden by the envelope")
    panel_labels(axes); save(fig,out_dir,"part2-reference-amplitude-phase")


def part2_frontier(data: dict, out_dir: Path) -> None:
    f=data["frontier"]; t=1000*np.asarray(f["T_grid"]); floor=1e-13
    y2=np.maximum(np.asarray(f["infid_m2"]),floor); y3=np.maximum(np.asarray(f["infid_m3"]),floor)
    fig,ax=plt.subplots(figsize=(7.7,3.85),constrained_layout=True)
    ax.semilogy(t,y2,"o-",color=COLORS["blue"],label=r"$\Omega_x,\Omega_y$")
    ax.semilogy(t,y3,"s-",color=COLORS["green"],label=r"$\Omega_x,\Omega_y,\Delta$")
    zero_t=t[(np.asarray(f["infid_m2"])<=0)|(np.asarray(f["infid_m3"])<=0)]
    if len(zero_t): ax.scatter(zero_t,np.full(len(zero_t),floor),marker="v",facecolor="white",edgecolor=COLORS["ink"],zorder=4,label=r"reported zero: $\leq10^{-13}$")
    ax.axvline(605.7437,color=COLORS["gray"],ls="--",lw=1,label="ideal-blockade QSL")
    ax.set(xlabel="duration [ns]",ylabel=r"$1-F_{\mathrm{tr}}$",title="Saved finite-blockade duration sweep"); ax.legend(fontsize=7.5,ncol=2)
    save(fig,out_dir,"part2-duration-frontier")


def part3_results(g3:dict,g4:dict,out_dir:Path)->None:
    floors=sorted((float(k),v) for k,v in g3["mintime"].items()); target=np.array([x[0] for x in floors]); duration=np.array([x[1]["T_us"]*1000 for x in floors])
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.55),constrained_layout=True)
    axes[0].plot(1-target,duration,"o-",color=COLORS["red"])
    for x,y,r in zip(1-target,duration,[v["F_rollout"] for _,v in floors]): axes[0].annotate(f"rollout {r:.5f}",(x,y),xytext=(5,5),textcoords="offset points",fontsize=7)
    axes[0].set_xscale("log"); axes[0].invert_xaxis(); axes[0].set(xlabel="allowed trace infidelity",ylabel="duration [ns]",title="Collocation fidelity floors")
    bell=g4["bell"]; time=np.linspace(0,bell["T_us"],bell["N"])
    axes[1].plot(time,np.asarray(bell["omega"])/(2*np.pi),color=COLORS["sky"],label=r"$\Omega/2\pi$"); axes[1].plot(time,np.asarray(bell["delta"])/(2*np.pi),color=COLORS["green"],label=r"$\Delta/2\pi$")
    axes[1].set(xlabel=r"time [$\mu$s]",ylabel="MHz",title=f"Piccolo Bell trajectory · F={bell['F_rollout']:.6f}"); axes[1].legend(fontsize=7.5)
    panel_labels(axes); save(fig,out_dir,"part3-collocation-piccolo-results")


def part3_constraints(g1:dict,g3:dict,out_dir:Path)->None:
    r=g3["mintime"]["0.9999"]; x=np.asarray(r["omega_x"]); y=np.asarray(r["omega_y"]); t=1000*np.linspace(0,r["T_us"],len(x)); amp=np.hypot(x,y); dt=r["dt_us"]
    slew=np.hypot(np.diff(x),np.diff(y))/dt; curvature=np.hypot(np.diff(x,2),np.diff(y,2))/(dt**2)
    fig,axes=plt.subplots(3,1,figsize=(8.2,6.0),constrained_layout=True,sharex=True)
    series=((t,amp/g1["constants"]["omega_max"],"Amplitude / bound",COLORS["red"]),(t[1:],slew/148,"Slew / bound",COLORS["blue"]),(t[1:-1],curvature/1739,"Curvature / bound",COLORS["green"]))
    for ax,(tx,value,title,color) in zip(axes,series):
        ax.plot(tx,value,color=color); ax.axhline(1,color=COLORS["gray"],ls="--",lw=1,label="constraint"); ax.fill_between(tx,0,value,color=color,alpha=.08); ax.set(ylabel="fraction",title=title,ylim=(0,1.08))
    axes[0].legend(fontsize=7.4,loc="lower right"); axes[2].set(xlabel="time [ns]"); axes[0].set_xlim(t[0],t[-1]); panel_labels(axes); save(fig,out_dir,"part3-active-constraints")


def part3_gauge_hole(out_dir:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.55),constrained_layout=True)
    ax=axes[0]; ax.grid(False); phases=[0,.58,.82,2.0]; labels=[r"$\phi_{00}$",r"$\phi_{01}$",r"$\phi_{10}$",r"$\phi_{11}$"]
    for i,(phase,label) in enumerate(zip(phases,labels)):
        ax.arrow(0,0,.8*np.cos(phase),.8*np.sin(phase),width=.012,head_width=.07,color=COLORS["blue"] if i<3 else COLORS["red"],length_includes_head=True); ax.text(.92*np.cos(phase),.92*np.sin(phase),label,ha="center",va="center")
    ax.add_patch(mpl.patches.Circle((0,0),.8,fill=False,ec=COLORS["light"])); ax.set(xlim=(-1.15,1.15),ylim=(-1.05,1.15),aspect="equal",title="Population return does not fix phase"); ax.axis("off")
    p=np.linspace(-np.pi,np.pi,300); ax=axes[1]; ax.plot(p,np.sin(p/2)**2,color=COLORS["purple"]); ax.axvline(np.pi,color=COLORS["gray"],ls="--"); ax.set(xlabel=r"invariant entangling phase $\Phi$",ylabel="phase-aware CZ score",title=r"$\Phi=\phi_{00}-\phi_{01}-\phi_{10}+\phi_{11}$"); ax.set_xticks([-np.pi,0,np.pi],[r"$-\pi$","0",r"$\pi$"])
    panel_labels(axes); save(fig,out_dir,"part3-gauge-hole")


def part4_noise(scores:dict,out_dir:Path)->None:
    selected=["CRAB","open-system GRAPE","collocation 0.9999","robust GRAPE","GRAPE","Krotov","Levine–Pichler"]
    rows=[next(r for r in scores["scores"] if r["name"]==n) for n in selected]
    exposure=np.array([r["rydberg_exposure_us"] for r in rows]); infid=1-np.array([r["average_fidelity_unconditional"] for r in rows]); excess=infid-np.array([r["coherent_trace_infidelity"] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.05),constrained_layout=True)
    offsets={"CRAB":(5,8),"open-system GRAPE":(5,-13),"collocation 0.9999":(5,-13),"robust GRAPE":(5,8),"GRAPE":(5,-13),"Krotov":(-52,8),"Levine–Pichler":(5,-14)}
    for r,x,y in zip(rows,exposure,excess):
        color=METHOD_COLORS.get(r["name"],COLORS["gray"]); axes[0].scatter(x,y,s=50,color=color,edgecolor="white",zorder=3); axes[0].annotate(r["name"],(x,y),xytext=offsets[r["name"]],textcoords="offset points",fontsize=6.8)
    kappa=float(np.dot(exposure,excess)/np.dot(exposure,exposure)); grid=np.linspace(0,exposure.max()*1.04,100); axes[0].plot(grid,kappa*grid,"--",color=COLORS["ink"],lw=1.1,label=fr"fit: $\kappa={kappa:.3f}\,\mu\mathrm{{s}}^{{-1}}$"); axes[0].legend(fontsize=7.2)
    axes[0].set(xlabel=r"Rydberg exposure [$\mu$s]",ylabel="dissipative excess infidelity",title="Exposure remains a useful first-order budget")
    order=np.argsort(infid); y=np.arange(len(rows)); axes[1].barh(y,infid[order],color=[METHOD_COLORS.get(rows[i]["name"],COLORS["gray"]) for i in order]); axes[1].set_yticks(y,[rows[i]["name"] for i in order],fontsize=7.1); axes[1].invert_yaxis(); axes[1].set(xlabel="unconditional average infidelity",title="One leakage-aware ranking")
    xmax=max(infid)*1.24; axes[1].set_xlim(0,xmax)
    for yi,val in zip(y,infid[order]): axes[1].text(val+xmax*.015,yi,f"{val:.4f}",va="center",fontsize=7)
    panel_labels(axes); save(fig,out_dir,"part4-noise-exposure-ranking")


def part4_amplitude_response(scores:dict,out_dir:Path)->None:
    fig,ax=plt.subplots(figsize=(7.7,3.85),constrained_layout=True)
    for name in ("GRAPE","robust GRAPE"):
        r=scores["amplitude_error_response"][name]; y=np.maximum(np.asarray(r["trace_infidelity"]),1e-13)
        ax.semilogy(100*np.asarray(r["epsilon"]),y,label="nominal GRAPE" if name=="GRAPE" else "ensemble-robust GRAPE",color=METHOD_COLORS[name])
    ax.axvspan(-2,2,color=COLORS["gray"],alpha=.12,label="robust training range"); ax.set(xlabel="multiplicative amplitude error [%]",ylabel=r"$1-F_{\mathrm{tr}}$",title="Robustness is a response curve, not one nominal number"); ax.legend(fontsize=7.5)
    save(fig,out_dir,"part4-amplitude-error-response")


def part5_hardware(waveforms:dict,bridge:dict,out_dir:Path)->None:
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.0),constrained_layout=True)
    sample=waveforms["meta"]["sample_period_ns"]
    for row,(key,spacing) in enumerate((("r5p0","5.0"),("r6p5","6.5"))):
        item=waveforms[key]
        for col,control in enumerate(("omega_rad_per_us","delta_rad_per_us")):
            ax=axes[row,col]; nominal=np.asarray(item["nominal"][control])/(2*np.pi); delivered=np.asarray(item["delivered"][control])/(2*np.pi)
            tn=np.arange(len(nominal))*sample; td=np.arange(len(delivered))*sample
            ax.plot(tn,nominal,color=COLORS["blue"],alpha=.82,label="programmed samples"); ax.plot(td,delivered,color=COLORS["red"],label="delivered model")
            ax.axvspan(tn[-1],td[-1],color=COLORS["orange"],alpha=.10,label="modulation tail" if row==0 and col==0 else None)
            ax.set(ylabel="MHz",title=(r"$\Omega/2\pi$" if col==0 else r"$\Delta/2\pi$")+f" · {spacing} µm")
        f0=bridge["bell_bridge"][spacing]["F_sequence_nominal"]; f1=bridge["bell_bridge"][spacing]["F_sequence_modulated"]
        axes[row,0].text(.02,.05,f"Bell F {f0:.6f} → {f1:.6f}",transform=axes[row,0].transAxes,fontsize=7.2,bbox={"facecolor":"white","edgecolor":COLORS["light"],"pad":2})
    axes[0,0].legend(ncol=2,fontsize=6.9,loc="upper right"); axes[-1,0].set_xlabel("actual sample time [ns]"); axes[-1,1].set_xlabel("actual sample time [ns]"); panel_labels(axes)
    save(fig,out_dir,"part5-programmed-delivered")


def part5_geometry(g1:dict,out_dir:Path)->None:
    r=np.array([5.0,6.5,8.66]); v=g1["constants"]["C6_rad_per_us_um6"]/r**6; ratio=v/(np.sqrt(2)*g1["constants"]["omega_max"])
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.55),constrained_layout=True)
    bars=axes[0].bar(["5.0","6.5","8.66"],v/(2*np.pi),color=[COLORS["green"],COLORS["orange"],COLORS["purple"]]); axes[0].set(xlabel="spacing [µm]",ylabel=r"$V/2\pi$ [MHz]",title=r"Geometry changes $V=C_6/R^6$")
    for b,val in zip(bars,v/(2*np.pi)): axes[0].text(b.get_x()+b.get_width()/2,val+max(v/(2*np.pi))*.025,f"{val:.2f}",ha="center",fontsize=7.5)
    axes[1].plot(r,ratio,"o-",color=COLORS["blue"]); axes[1].axhline(1,color=COLORS["gray"],ls="--",lw=1); axes[1].set(xlabel="spacing [µm]",ylabel=r"$V/(\sqrt{2}\Omega_{\max})$",title="Finite-blockade authority")
    for x,y in zip(r,ratio): axes[1].annotate(f"{y:.2f}",(x,y),xytext=(4,5),textcoords="offset points",fontsize=7.5)
    panel_labels(axes); save(fig,out_dir,"part5-geometry-blockade")


def part5_spam(data:dict,out_dir:Path)->None:
    spam=data["meta"]["calibration"]["fitted_spam"]; labels=["false positive","false negative","state prep."]; values=np.array([spam["p_false_pos"],spam["p_false_neg"],spam["state_prep_error"]])
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.5),constrained_layout=True)
    bars=axes[0].bar(labels,100*values,color=[COLORS["orange"],COLORS["red"],COLORS["sky"]]); axes[0].set(ylabel="fitted probability [%]",title="Anonymized aggregate SPAM fit")
    for b,val in zip(bars,values): axes[0].text(b.get_x()+b.get_width()/2,100*val+.18,f"{100*val:.2f}%",ha="center",fontsize=7.5)
    predicted=data["run"]["predicted_P_Bell"]["M4_calibrated"]; axes[1].bar(["ideal target","calibrated\nprediction"],[1,predicted],color=[COLORS["green"],COLORS["purple"]],width=.62); axes[1].set_ylim(.84,1.015); axes[1].set(ylabel=r"predicted $P_{\mathrm{Bell}}$",title="The 1,000-shot item is a proposal"); axes[1].text(1,predicted+.006,f"{predicted:.3f}",ha="center",fontsize=8)
    panel_labels(axes); save(fig,out_dir,"part5-spam-run-plan")


def validate(g1:dict,g2:dict,g3:dict,g4:dict,scores:dict,g7:dict)->None:
    hx,hy,hd,h0=hamiltonian_parts(g1)
    for h in (hx,hy,hd,h0): assert np.max(abs(h-h.conj().T))<1e-14
    assert np.isclose(g1["constants"]["V_canonical"],55.40627328)
    assert g2["grape_cz"]["N"]==len(g2["grape_cz"]["omega_x"])==100
    assert g3["mintime"]["0.9999"]["F_rollout"]>=.9999 and g4["meta"]["piccolo_version"]=="1.19.0"
    for r in scores["scores"]: assert np.isclose(r["average_fidelity_unconditional"],(4*r["process_fidelity"]+r["survival"])/5)
    serialized=json.dumps(g7).lower(); assert all(x not in serialized for x in ("fresnel","c20efe8f","a0be90ab","team5","account")); assert g7["submission_status"].startswith("proposal only")


def main()->None:
    parser=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir",type=Path,default=root/"assets/data/neutral-atom-control"); parser.add_argument("--output-dir",type=Path,default=root/"assets/img/neutral-atom-control"); args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); setup_style()
    g1=load(args.data_dir/"g1_baselines.json"); g2=load(args.data_dir/"g2_pulses.json"); g3=load(args.data_dir/"g3_collocation.json"); g4=load(args.data_dir/"g4_piccolo_bell.json"); g6=load(args.data_dir/"g6_bridge.json"); wave=load(args.data_dir/"g6_delivered_waveforms.json"); g7=load(args.data_dir/"g7_qpu_runplan_sanitized.json"); jp=load(args.data_dir/"jp_figshare_cz.json"); rough=load(args.data_dir/"pulse_smoothness_audit.json"); scores=load(args.data_dir/"g5_leakage_aware_scores.json"); validate(g1,g2,g3,g4,scores,g7)
    part1_baselines(g1,args.output_dir); part1_blockade_population(g1,args.output_dir); part2_pulses(g2,args.output_dir); part2_reference(g2,jp,rough,args.output_dir); part2_frontier(g2,args.output_dir); part3_results(g3,g4,args.output_dir); part3_constraints(g1,g3,args.output_dir); part3_gauge_hole(args.output_dir); part4_noise(scores,args.output_dir); part4_amplitude_response(scores,args.output_dir); part5_hardware(wave,g6,args.output_dir); part5_geometry(g1,args.output_dir); part5_spam(g7,args.output_dir)
    print(f"wrote 13 SVG masters and 13 PNG displays to {args.output_dir}; cover is built from TikZ")


if __name__=="__main__": main()
