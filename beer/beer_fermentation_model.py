#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
啤酒工业发酵过程动态建模与灵敏度分析
==========================================

本脚本属于化工教学课件的一部分，针对下面发酵（拉格啤酒 Lager）主发酵过程，
演示"化工三传一反"中的反应动力学建模与"控制工程"中的灵敏度分析方法。

------------------------------------------------------------------------------
一、数学模型（ODE 系统）
------------------------------------------------------------------------------
记号：
    X : 酵母细胞浓度 (g/L)
    S : 底物（可发酵糖）浓度 (g/L)
    P : 产物（乙醇）浓度 (g/L)
    CO2 : 累积 CO2 释放量 (g/L)
    T : 发酵温度 (K)

1) 比生长速率（Monod 方程 + Luong 产物抑制模型）
       μ = μmax · S/(Ks + S) · (1 - P/Pmax)^n          若 P < Pmax，否则 μ = 0

   其中 μmax 为最大比生长速率，Ks 为 Monod 半饱和常数，
   Pmax 为酵母最大耐受乙醇浓度，n 为产物抑制指数（一般 n=1，本脚本 n=1）。

2) 细胞生长（含衰亡项）
       dX/dt = μ · X - kd · X

3) 底物消耗（Luedeking-Piret 形式，区分生长相关与维持相关）
       dS/dt = -(1/Yx_s) · μ · X - m_s · X

4) 产物生成（Luedeking-Piret 模型）
       dP/dt = (α · μ + β) · X
   其中 α 为生长相关产物得率系数，β 为非生长相关（维持）产物得率系数。

5) CO2 生成（与乙醇生成化学计量相关）
       dCO2/dt = Yp_co2 · dP/dt
   发酵反应 C6H12O6 -> 2 C2H5OH + 2 CO2，理论 Yp_co2 = 88/92 ≈ 0.957。

6) 温度对速率的影响（Arrhenius 修正）
       f(T) = exp[-Ea/R · (1/T - 1/Tref)]
   其中 Ea 为活化能，R 为气体常数，Tref 为参考温度。
   修正后的比生长速率 μmax_eff = μmax_ref · f(T)。

------------------------------------------------------------------------------
二、参数取值与文献来源
------------------------------------------------------------------------------
参数取工业拉格啤酒 10°C 主发酵的典型值，并附文献参考：

* μmax = 0.04 h^-1      —— 10°C 拉格发酵比生长速率典型值
                           (参考: Gevertz et al., 1995; Brányik et al., 2005
                                  J. Inst. Brew. 111:122-134)
* Ks = 0.5 g/L           —— Monod 半饱和常数，糖代谢常见 0.1~1 g/L
* Pmax = 80 g/L          —— Luong 模型最大耐受乙醇浓度（酿酒酵母约 80~110 g/L）
                           (参考: Luong JHT, 1985, Biotechnol Bioeng 27:280-285)
* n = 1                  —— Luong 抑制指数
* Yx/s = 0.14            —— 菌体对底物得率（g/g），啤酒酵母典型 0.10~0.18
* m_s = 0.01 g/(g·h)     —— 维持系数
* α = 3.8                —— 生长相关产物得率系数（g 乙醇 / g 菌体）
* β = 0.02 h^-1          —— 非生长相关产物得率系数
* kd = 0.001 h^-1        —— 衰亡系数
* Yp_co2 = 0.957         —— CO2/乙醇化学计量比（理论值）
* Ea = 50000 J/mol       —— 综合表观活化能（生物反应典型 40~80 kJ/mol）
* Tref = 283.15 K (10°C) —— μmax 所对应的参考温度
* R  = 8.314 J/(mol·K)   —— 气体常数

参考文献：
    [1] Monod J. (1949) The growth of bacterial cultures. Ann Rev Microbiol 3:371-394.
    [2] Luong JHT. (1985) Kinetics of ethanol inhibition in alcohol fermentation.
        Biotechnol Bioeng 27:280-285.
    [3] Luedeking R, Piret EL. (1959) A kinetic study of the lactic acid fermentation.
        J Biochem Microbial Technol Eng 1:393-412.
    [4] Brányik T, Vicente AA, Dostálek P, Teixeira JA. (2005)
        A review of flavour formation in continuous beer fermentation.
        J Inst Brew 111:122-134.

初值（工业拉格 12°P 麦汁）：
    X0  = 2 g/L     （接种后酵母浓度）
    S0  = 120 g/L   （约 12°P 麦汁可发酵糖）
    P0  = 0 g/L
    CO2_0 = 0 g/L

发酵条件：主发酵 10°C 恒温，发酵时间 0~168 h（7 天）。

------------------------------------------------------------------------------
三、数值求解
------------------------------------------------------------------------------
采用 scipy.integrate.solve_ivp 显式 Runge-Kutta（RK45）方法。

------------------------------------------------------------------------------
四、灵敏度分析
------------------------------------------------------------------------------
对参数 μmax、Ks、Pmax、Yx/s、温度 T 各取 -50%/-20%/+20%/+50% 五个水平
（温度以 ±2°C / ±5°C 对应），扫描各参数变化对：
    (1) 最终乙醇浓度 P_end
    (2) 残糖 S_end
    (3) 最大细胞浓度 X_max
    (4) 发酵完成时间 t_done（S 降至阈值 2 g/L 的时间）
的影响，输出相对灵敏度表与 Tornado 图。

------------------------------------------------------------------------------
五、输出
------------------------------------------------------------------------------
- 控制台打印关键结果与灵敏度表
- 保存 5 张图到 /workspace/figures/
------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")  # 无显示环境也保存图片
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ==========================================================================
# 全局参数设置与字体配置
# ==========================================================================
def _setup_matplotlib() -> None:
    """配置 matplotlib 中文字体与学术风格。

    策略：优先使用系统中可用的 CJK 字体（Noto Sans CJK SC 等），
    若均不可用则回退到英文标签（在绘图函数中通过 USE_CN 控制）。
    """
    # 候选中文字体（按优先级）
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",  # CJK 公共字符覆盖
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    from matplotlib.font_manager import fontManager

    available = {f.name for f in fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen is not None:
        rcParams["font.family"] = chosen
        rcParams["axes.unicode_minus"] = False
        rcParams["font.sans-serif"] = [chosen]
        globals()["USE_CN"] = True
        globals()["CJK_FONT"] = chosen
    else:
        # 无中文字体，回退英文标签
        globals()["USE_CN"] = False
        globals()["CJK_FONT"] = None


# 中文/英文标签词典
LABEL = {
    "time":     ("时间 / h",                    "Time / h"),
    "X":        ("酵母细胞浓度 X / (g·L$^{-1}$)", "Yeast X / (g/L)"),
    "S":        ("糖浓度 S / (g·L$^{-1}$)",      "Sugar S / (g/L)"),
    "P":        ("乙醇浓度 P / (g·L$^{-1}$)",     "Ethanol P / (g/L)"),
    "CO2":      ("CO$_2$ 释放量 / (g·L$^{-1}$)", "CO2 released / (g/L)"),
    "X_title":  ("酵母细胞浓度",                  "Yeast cell concentration"),
    "S_title":  ("底物（糖）浓度",                "Substrate (sugar) concentration"),
    "P_title":  ("乙醇产物浓度",                  "Ethanol product concentration"),
    "CO2_title":("CO$_2$ 累积释放量",            "Cumulative CO2 release"),
    "fig1":     ("啤酒主发酵动态曲线（10°C 拉格）",
                 "Lager main fermentation dynamics (10°C)"),
    "fig2":     ("μmax 对乙醇生成曲线 P(t) 的影响",
                 "Effect of μmax on ethanol curve P(t)"),
    "fig3":     ("发酵温度对乙醇生成曲线 P(t) 的影响",
                 "Effect of temperature on ethanol curve P(t)"),
    "fig4":     ("各参数对最终乙醇浓度 P_end 的灵敏度（Tornado 图）",
                 "Sensitivity of P_end to parameters (Tornado)"),
    "fig5":     ("乙醇-底物相图 P vs S",         "Phase plot P vs S"),
    "P_end":    ("最终乙醇浓度",                 "Final ethanol P_end"),
    "S_end":    ("残糖",                         "Residual sugar"),
    "X_max":    ("最大细胞浓度",                 "Max cell conc."),
    "t_done":   ("发酵完成时间",                 "Fermentation done time"),
    "rel":      ("相对变化率 / %",               "Relative change / %"),
}


def L(key: str) -> str:
    """根据是否启用中文返回对应标签。"""
    return LABEL[key][0 if USE_CN else 1]


# ==========================================================================
# 模型参数
# ==========================================================================
@dataclass
class Params:
    """发酵动力学参数集合。"""
    # 动力学参数
    mu_max: float = 0.04       # h^-1，最大比生长速率（参考温度下）
    Ks: float = 0.5           # g/L，Monod 半饱和常数
    Pmax: float = 80.0         # g/L，Luong 最大耐受乙醇浓度
    n: float = 1.0             # Luong 抑制指数
    Yx_s: float = 0.14         # g/g，菌体对底物得率
    m_s: float = 0.01          # g/(g·h)，维持系数
    alpha: float = 3.8         # 生长相关产物得率
    beta: float = 0.02          # h^-1，非生长相关产物得率
    kd: float = 0.001          # h^-1，衰亡系数
    Yp_co2: float = 0.957      # CO2 / 乙醇化学计量比
    # Arrhenius
    Ea: float = 50000.0        # J/mol，表观活化能
    R: float = 8.314           # J/(mol·K)
    Tref: float = 283.15       # K（10°C）
    # 操作条件
    T: float = 283.15          # K，发酵温度
    # 初值
    X0: float = 2.0            # g/L
    S0: float = 120.0          # g/L
    P0: float = 0.0            # g/L
    CO2_0: float = 0.0         # g/L
    # 发酵时间
    t_end: float = 168.0       # h

    def copy(self) -> "Params":
        return Params(**self.__dict__)


# ==========================================================================
# 模型 ODE
# ==========================================================================
def arrhenius_factor(T: float, Tref: float, Ea: float, R: float = 8.314) -> float:
    """Arrhenius 温度修正因子 f(T) = exp[-Ea/R·(1/T - 1/Tref)]。

    T = Tref 时 f = 1；温度升高 f 增大（速率加快）。

    Parameters
    ----------
    T : float
        当前温度 (K)。
    Tref : float
        参考温度 (K)。
    Ea : float
        表观活化能 (J/mol)。
    R : float
        气体常数 (J/(mol·K))。

    Returns
    -------
    float
        无量纲修正因子。
    """
    if T <= 0:
        return 0.0
    return float(np.exp(-Ea / R * (1.0 / T - 1.0 / Tref)))


def monod_luong_mu(S: float, P: float, p: Params) -> float:
    """Monod + Luong 比生长速率 μ。

    μ = μmax_eff · S/(Ks+S) · (1 - P/Pmax)^n   (P < Pmax)
    μ = 0                                      (P ≥ Pmax)

    其中 μmax_eff = μmax · f(T)。
    """
    mu_max_eff = p.mu_max * arrhenius_factor(p.T, p.Tref, p.Ea, p.R)
    substrate_term = S / (p.Ks + S) if (p.Ks + S) > 0 else 0.0
    if P >= p.Pmax:
        inhibition = 0.0
    else:
        inhibition = (1.0 - P / p.Pmax) ** p.n
    return mu_max_eff * substrate_term * inhibition


def model_odes(t: float, y: np.ndarray, p: Params) -> List[float]:
    """发酵动力学 ODE 系统。

    Parameters
    ----------
    t : float
        时间 (h)，本系统自治（不显式含 t）。
    y : np.ndarray
        状态向量 [X, S, P, CO2]。
    p : Params
        参数集合。

    Returns
    -------
    List[float]
        导数 [dX/dt, dS/dt, dP/dt, dCO2/dt]。
    """
    X, S, P, CO2 = y
    # 数值保护（防止积分器越界导致负浓度）
    X = max(X, 0.0)
    S = max(S, 0.0)
    P = max(P, 0.0)
    CO2 = max(CO2, 0.0)

    # 底物耗尽时糖消耗项应停止（避免 S 出现负值）
    # 维持项 m_s·X 也需在 S=0 时停止
    S_available = 1.0 if S > 1e-6 else 0.0

    mu = monod_luong_mu(S, P, p)
    # 底物耗尽时生长停止
    mu = mu * S_available

    dXdt = mu * X - p.kd * X
    dSdt = -(1.0 / p.Yx_s) * mu * X - p.m_s * X * S_available
    dPdt = (p.alpha * mu + p.beta) * X
    dCO2dt = p.Yp_co2 * dPdt

    return [dXdt, dSdt, dPdt, dCO2dt]


# ==========================================================================
# 数值求解
# ==========================================================================
def simulate(p: Params, n_points: int = 500) -> Dict[str, np.ndarray]:
    """数值积分发酵 ODE 系统。

    Parameters
    ----------
    p : Params
        参数。
    n_points : int
        输出时间点数。

    Returns
    -------
    dict
        包含 't','X','S','P','CO2' 数组。
    """
    y0 = [p.X0, p.S0, p.P0, p.CO2_0]
    t_eval = np.linspace(0, p.t_end, n_points)
    sol = solve_ivp(
        fun=model_odes,
        t_span=(0.0, p.t_end),
        y0=y0,
        args=(p,),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(f"ODE 积分失败: {sol.message}")
    return {
        "t": sol.t,
        "X": sol.y[0],
        "S": sol.y[1],
        "P": sol.y[2],
        "CO2": sol.y[3],
    }


# ==========================================================================
# 关键指标
# ==========================================================================
def key_metrics(sol: Dict[str, np.ndarray],
                S_threshold: float = 2.0) -> Dict[str, float]:
    """从解中提取关键指标。

    Parameters
    ----------
    sol : dict
        simulate() 返回的结果。
    S_threshold : float
        定义发酵完成的残糖阈值 (g/L)。

    Returns
    -------
    dict
        包含 P_end, S_end, X_end, X_max, t_done 等指标。
    """
    t = sol["t"]
    X = sol["X"]
    S = sol["S"]
    P = sol["P"]

    result = {
        "P_end": float(P[-1]),
        "S_end": float(S[-1]),
        "X_end": float(X[-1]),
        "X_max": float(np.max(X)),
        "P_max": float(np.max(P)),
        "t_Xmax": float(t[int(np.argmax(X))]),
    }
    # 发酵完成时间：S 降至阈值以下的首个时刻
    idx = np.where(S <= S_threshold)[0]
    if len(idx) > 0:
        result["t_done"] = float(t[idx[0]])
        result["reached_threshold"] = True
    else:
        result["t_done"] = float("nan")
        result["reached_threshold"] = False
    return result


# ==========================================================================
# 灵敏度分析
# ==========================================================================
def sensitivity_analysis(base: Params,
                         S_threshold: float = 2.0
                         ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """对 μmax、Ks、Pmax、Yx/s、T 做扇形扫描灵敏度分析。

    每个参数取相对基准 -50%/-20%/+20%/+50% 共 4 个扰动水平
    （温度 T 改为 ±2°C / ±5°C 的绝对扰动，等效于对 283.15K 的 ~0.7%/1.8%）。

    Parameters
    ----------
    base : Params
        基准参数集。
    S_threshold : float
        发酵完成残糖阈值。

    Returns
    -------
    sweep_results : dict
        每个 参数 -> {水平标签: {指标}} 的嵌套字典。
    base_metrics : dict
        基准指标。
    """
    base_sol = simulate(base)
    base_metrics = key_metrics(base_sol, S_threshold)

    # 扰动参数定义：键名 -> (取值函数: 比例/绝对 -> 新值)
    # 比例扰动列表（对基准值的相对变化）
    ratios = [0.5, 0.8, 1.2, 1.5]
    ratio_labels = ["-50%", "-20%", "+20%", "+50%"]

    # 温度的绝对扰动（°C）
    temp_offsets_C = [-5.0, -2.0, 2.0, 5.0]

    sweep_results: Dict[str, Dict] = {}

    # ---- 比例扰动参数 ----
    proportional = {
        "mu_max": ("μmax", lambda p, r: p.mu_max * r),
        "Ks":     ("Ks",   lambda p, r: p.Ks * r),
        "Pmax":   ("Pmax", lambda p, r: p.Pmax * r),
        "Yx_s":   ("Yx/s", lambda p, r: p.Yx_s * r),
    }

    for key, (disp, mutate) in proportional.items():
        sweep_results[key] = {"display": disp, "levels": {}}
        for r, lab in zip(ratios, ratio_labels):
            p = base.copy()
            new_val = mutate(p, r)
            setattr(p, key, new_val)
            sol = simulate(p)
            m = key_metrics(sol, S_threshold)
            sweep_results[key]["levels"][lab] = {
                "param_value": float(new_val),
                **m,
            }

    # ---- 温度扰动（绝对 °C） ----
    sweep_results["T"] = {"display": "T (°C)", "levels": {}}
    for off, lab in zip(temp_offsets_C, ratio_labels):
        p = base.copy()
        new_T_K = base.T + off  # K，因为 base.T 是 K
        p.T = new_T_K
        sol = simulate(p)
        m = key_metrics(sol, S_threshold)
        sweep_results["T"]["levels"][lab] = {
            "param_value": float(new_T_K - 273.15),  # 显示 °C
            **m,
        }

    return sweep_results, base_metrics


def relative_change(new_val: float, base_val: float) -> float:
    """相对变化率 (%)。基准值为零时返回 NaN。"""
    if base_val == 0:
        return float("nan")
    return (new_val - base_val) / base_val * 100.0


def fmt_rel(val: float) -> str:
    """格式化相对变化率字符串；NaN 显示 N/A。"""
    if val != val:  # NaN
        return "    N/A    "
    return f"{val:>+9.2f}%"


# ==========================================================================
# 可视化
# ==========================================================================
def plot_dynamics(sol: Dict[str, np.ndarray], p: Params,
                  outpath: str) -> None:
    """fig1: X, S, P, CO2 四条曲线随时间变化（双纵轴）。"""
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    t = sol["t"]
    color_X = "#2ca02c"
    color_S = "#1f77b4"
    color_P = "#d62728"
    color_C = "#7f7f7f"

    l_X, = ax1.plot(t, sol["X"], color=color_X, lw=2.0, label=L("X"))
    l_S, = ax1.plot(t, sol["S"], color=color_S, lw=2.0, label=L("S"))
    ax1.set_xlabel(L("time"))
    ax1.set_ylabel(f"{L('X')}  /  {L('S')}", color="#444")
    ax1.tick_params(axis="y", labelcolor="#444")

    ax2 = ax1.twinx()
    l_P, = ax2.plot(t, sol["P"], color=color_P, lw=2.0, label=L("P"))
    l_C, = ax2.plot(t, sol["CO2"], color=color_C, lw=1.6, ls="--",
                    label=L("CO2"))
    ax2.set_ylabel(f"{L('P')}  /  {L('CO2')}", color="#444")

    # 合并图例
    lines = [l_X, l_S, l_P, l_C]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)

    ax1.grid(True, ls=":", alpha=0.6)
    T_C = p.T - 273.15
    fig.suptitle(f"{L('fig1')}   (T = {T_C:.0f}°C, t = 0–{p.t_end:.0f} h)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_sensitivity_curve(param_key: str,
                            param_display: str,
                            levels: List[Tuple[str, float, np.ndarray]],
                            base_t: np.ndarray,
                            base_P: np.ndarray,
                            title: str,
                            outpath: str,
                            xlabel_override: str | None = None
                            ) -> None:
    """通用：单参数变化下 P(t) 多条曲线。"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # 基准曲线
    ax.plot(base_t, base_P, color="black", lw=2.2, label="基准 / Base",
            zorder=5)
    cmap = plt.cm.viridis
    n = max(len(levels), 1)
    for i, (lab, val, P_curve) in enumerate(levels):
        ax.plot(base_t, P_curve, color=cmap(i / max(n - 1, 1)), lw=1.6,
                label=f"{lab} ({val:.3g})")
    ax.set_xlabel(L("time"))
    ax.set_ylabel(L("P"))
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_sensitivity_mu_max(base: Params, outpath: str) -> None:
    """fig2: μmax 变化对 P(t) 影响。"""
    ratios = [0.5, 0.8, 1.2, 1.5]
    labels = ["-50%", "-20%", "+20%", "+50%"]
    base_sol = simulate(base)
    levels = []
    for r, lab in zip(ratios, labels):
        p = base.copy()
        p.mu_max = base.mu_max * r
        sol = simulate(p)
        levels.append((lab, p.mu_max, sol["P"]))
    plot_sensitivity_curve(
        "mu_max", "μmax", levels,
        base_sol["t"], base_sol["P"],
        L("fig2"), outpath,
    )


def plot_sensitivity_temperature(base: Params, outpath: str) -> None:
    """fig3: 温度变化对 P(t) 影响。"""
    offsets = [-5.0, -2.0, 2.0, 5.0]
    labels = ["-50%", "-20%", "+20%", "+50%"]
    base_sol = simulate(base)
    levels = []
    for off, lab in zip(offsets, labels):
        p = base.copy()
        p.T = base.T + off
        T_C = p.T - 273.15
        sol = simulate(p)
        levels.append((lab, T_C, sol["P"]))
    plot_sensitivity_curve(
        "T", "T", levels,
        base_sol["t"], base_sol["P"],
        L("fig3"), outpath,
    )


def plot_tornado(sweep_results: Dict, base_metrics: Dict[str, float],
                outpath: str) -> None:
    """fig4: 各参数对 P_end 的灵敏度 Tornado 条形图。

    横轴为 P_end 相对基准的变化率 (%)，对每个参数画出
    -50%/-20%/+20%/+50% 四个水平对应的影响范围。
    """
    param_keys = ["mu_max", "Ks", "Pmax", "Yx_s", "T"]
    # 每个参数取最大正/负相对变化
    rows = []
    for k in param_keys:
        disp = sweep_results[k]["display"]
        levels = sweep_results[k]["levels"]
        rels = []
        for lab, m in levels.items():
            rel = relative_change(m["P_end"], base_metrics["P_end"])
            rels.append((lab, rel))
        # 取最负和最正
        rel_vals = [r for _, r in rels]
        min_rel = min(rel_vals)
        max_rel = max(rel_vals)
        rows.append((disp, min_rel, max_rel, rels))

    # 按"影响幅度"（max - min）排序
    rows.sort(key=lambda r: abs(r[2]) + abs(r[1]), reverse=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ypos = np.arange(len(rows))
    for i, (disp, min_rel, max_rel, rels) in enumerate(rows):
        # 主条形：从 0 到 min/max
        ax.barh(i, max_rel, color="#d62728", alpha=0.75,
                edgecolor="black", linewidth=0.5)
        ax.barh(i, min_rel, color="#1f77b4", alpha=0.75,
                edgecolor="black", linewidth=0.5)
        # 在条形端点标注数值
        ax.text(max_rel + 0.3, i, f"{max_rel:+.1f}%", va="center",
                fontsize=8, color="#d62728")
        ax.text(min_rel - 0.3, i, f"{min_rel:+.1f}%", va="center",
                ha="right", fontsize=8, color="#1f77b4")

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"{L('P_end')}  {L('rel')}")
    ax.set_title(L("fig4"), fontsize=12)
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_phase(sol: Dict[str, np.ndarray], outpath: str) -> None:
    """fig5: P vs S 相图。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    S = sol["S"]
    P = sol["P"]
    t = sol["t"]
    sc = ax.scatter(S, P, c=t, cmap="plasma", s=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(L("time"))
    # 标注起点与终点
    ax.scatter([S[0]], [P[0]], color="green", s=80, zorder=5,
               label=f"起点 (t=0): S={S[0]:.1f}, P={P[0]:.1f}")
    ax.scatter([S[-1]], [P[-1]], color="red", s=80, marker="*", zorder=5,
               label=f"终点 (t={t[-1]:.0f}h): S={S[-1]:.2f}, P={P[-1]:.2f}")
    ax.set_xlabel(L("S"))
    ax.set_ylabel(L("P"))
    ax.set_title(L("fig5"), fontsize=12)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ==========================================================================
# 控制台打印
# ==========================================================================
def print_key_results(base: Params, base_metrics: Dict[str, float]) -> None:
    print("=" * 64)
    print("啤酒主发酵基准模拟关键结果（10°C 拉格，168 h）")
    print("=" * 64)
    print(f"  最大比生长速率 μmax        = {base.mu_max:.4f} h^-1")
    print(f"  Arrhenius 温度修正因子 f(T) = {arrhenius_factor(base.T, base.Tref, base.Ea, base.R):.4f}")
    print(f"  发酵温度 T                  = {base.T - 273.15:.1f} °C")
    print("-" * 64)
    print(f"  最终乙醇浓度  P_end        = {base_metrics['P_end']:.3f} g/L")
    print(f"  残糖        S_end          = {base_metrics['S_end']:.3f} g/L")
    print(f"  终点细胞浓度 X_end         = {base_metrics['X_end']:.3f} g/L")
    print(f"  最大细胞浓度 X_max         = {base_metrics['X_max']:.3f} g/L  "
          f"(t = {base_metrics['t_Xmax']:.1f} h)")
    if base_metrics["reached_threshold"]:
        print(f"  发酵完成时间 t_done (S≤2)   = {base_metrics['t_done']:.2f} h")
    else:
        print(f"  发酵完成时间 t_done        = 未在 {base.t_end:.0f} h 内达到 S≤2 g/L")
    # 体积百分比估算（乙醇密度 0.789 g/mL）
    abv = base_metrics["P_end"] / 7.89
    print(f"  估算 ABV (体积百分比)       ≈ {abv:.2f} % v/v")
    print("=" * 64)


def print_sensitivity_table(sweep_results: Dict,
                            base_metrics: Dict[str, float]) -> None:
    print("\n")
    print("=" * 92)
    print("灵敏度分析表 —— 各参数变化对关键指标的相对影响 (%)")
    print("=" * 92)
    header = (f"{'参数':<10}{'水平':<8}{'取值':<14}"
              f"{'ΔP_end%':<12}{'ΔS_end(g/L)':<14}"
              f"{'ΔX_max%':<12}{'Δt_done%':<12}")
    print(header)
    print("-" * 92)
    for k in ["mu_max", "Ks", "Pmax", "Yx_s", "T"]:
        disp = sweep_results[k]["display"]
        levels = sweep_results[k]["levels"]
        for lab, m in levels.items():
            dP = relative_change(m["P_end"], base_metrics["P_end"])
            dX = relative_change(m["X_max"], base_metrics["X_max"])
            t_base = base_metrics["t_done"]
            t_new = m["t_done"]
            dt = relative_change(t_new, t_base) if (np.isfinite(t_base)
                                                    and np.isfinite(t_new)
                                                    and t_base > 0) else float("nan")
            # 残糖：基准接近 0 时显示绝对差（g/L），否则显示相对变化率
            if base_metrics["S_end"] > 1.0:  # 基准残糖大于 1 g/L 才用相对
                dS = relative_change(m["S_end"], base_metrics["S_end"])
                dS_str = fmt_rel(dS)
            else:
                dS_abs = m["S_end"] - base_metrics["S_end"]
                dS_str = f"{dS_abs:>+9.3f} "  # 绝对差 g/L
            pval = m["param_value"]
            # 温度显示 °C
            if k == "T":
                val_str = f"{pval:.2f}°C"
            else:
                val_str = f"{pval:.4g}"
            print(f"{disp:<10}{lab:<8}{val_str:<14}"
                  f"{fmt_rel(dP)}  " +
                  f"{dS_str}  " +
                  f"{fmt_rel(dX)}  " +
                  f"{fmt_rel(dt)}")
        print("-" * 92)
    print("说明：ΔP_end%、ΔX_max%、Δt_done% 为相对变化率 = (新-基准)/基准×100。")
    print("     ΔS_end(g/L) 为残糖绝对差（基准残糖 ≈ 0 时相对变化率无意义）。")
    print("     温度 T 的水平标签 -50%/-20%/+20%/+50% 实际对应 -5/-2/+2/+5 °C。")
    print("     Δt_done 出现 N/A 表示该工况在 168 h 内未达到 S≤2 g/L。")
    print("=" * 92)


# ==========================================================================
# 主函数
# ==========================================================================
def main() -> int:
    _setup_matplotlib()

    figures_dir = "/workspace/figures"
    os.makedirs(figures_dir, exist_ok=True)

    base = Params()
    print("正在求解基准发酵动力学 ODE 系统 ...")
    base_sol = simulate(base)
    base_metrics = key_metrics(base_sol)
    print_key_results(base, base_metrics)

    print("正在进行灵敏度扫描 ...")
    sweep_results, _ = sensitivity_analysis(base)
    print_sensitivity_table(sweep_results, base_metrics)

    # 绘图
    print("\n正在生成图表并保存至 %s ..." % figures_dir)
    plot_dynamics(base_sol, base,
                  os.path.join(figures_dir, "fig1_dynamics.png"))
    plot_sensitivity_mu_max(base,
                            os.path.join(figures_dir,
                                         "fig2_sensitivity_mu_max.png"))
    plot_sensitivity_temperature(
        base,
        os.path.join(figures_dir, "fig3_sensitivity_temperature.png"))
    plot_tornado(sweep_results, base_metrics,
                 os.path.join(figures_dir, "fig4_sensitivity_bar.png"))
    plot_phase(base_sol, os.path.join(figures_dir, "fig5_phase.png"))

    print("已保存以下图片：")
    for fn in ["fig1_dynamics.png", "fig2_sensitivity_mu_max.png",
               "fig3_sensitivity_temperature.png",
               "fig4_sensitivity_bar.png", "fig5_phase.png"]:
        full = os.path.join(figures_dir, fn)
        size = os.path.getsize(full) if os.path.exists(full) else -1
        print(f"  {full}   ({size} bytes)")

    print("\nCJK 字体使用情况: ",
          CJK_FONT if CJK_FONT else "无可用中文字体，使用英文标签")
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
