#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
速溶咖啡工业级萃取 · 化工三传一反模型

============================================================
配套课件：instant_coffee_courseware.html（第 4–6 章）
适用场景：化工原理教学、过程控制课程设计、研究生案例研究
============================================================

本脚本独立实现了课件第 4 章至第 6 章的全部数学模型：
  Part A — 动态萃取建模（一级动力学 ODE + 解析解）
  Part B — 灵敏度分析（弹性法）
  Part C — 稳态经济优化（约束 NLP 网格搜索）
  Part D — 可视化（matplotlib 四图）

运行方式：
  python3 instant_coffee_extraction_model.py
  或交互模式：python3 -c "from instant_coffee_extraction_model import *; demo()"

依赖：
  pip3 install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
#  全局样式（中文支持）
# ============================================================
rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================
#  Part A — 模型常数与核心函数
# ============================================================

Rg    = 8.314       # 气体常数  J/(mol·K)
Ea    = 52000        # 活化能  J/mol（咖啡可溶物溶出表观活化能）
Tref  = 363.15       # 参考温度  K（90 ℃）
k0    = 0.045        # 参考速率常数  1/min
dref  = 500          # 参考粒径  µm
Qref  = 12           # 参考流量  L/min
Xmax  = 1.0          # 可萃取上限（归一化）

def k_eff(T_celsius, d_p, Q):
    """
    有效速率常数 k(T, d_p, Q)
    封装阿伦尼乌斯（温度）· 内扩散（粒径²）· 对流传质（流量^0.6）
    """
    T = T_celsius + 273.15
    arr = np.exp((Ea / Rg) * (1.0 / Tref - 1.0 / T))
    d_factor = (dref / d_p) ** 2
    q_factor = (Q / Qref) ** 0.6
    return k0 * arr * d_factor * q_factor


def X_of_t(t, T, d_p, Q):
    """萃取率 X(t) 解析解：一级动力学指数饱和曲线"""
    k = k_eff(T, d_p, Q)
    return Xmax * (1.0 - np.exp(-k * t))


def t_95(T, d_p, Q):
    """达到 95% 萃取率所需时间"""
    return -np.log(0.05) / k_eff(T, d_p, Q)


def dP_ergun(d_p, Q, eps=0.4, mu=2.8e-4, L=1.0, A_bed=0.05):
    """
    Ergun 方程黏性项压降估算（示意量级）
    ΔP = 150 (1-ε)²/ε³ · μu/d_p² · L
    """
    dp_m = d_p * 1e-6                    # µm → m
    u = Q * 1e-3 / A_bed                  # L/min → m/s（示意）
    dP_Pa = 150 * (1 - eps)**2 / eps**3 * mu * u / dp_m**2 * L
    return dP_Pa / 1e5                     # Pa → bar


# ============================================================
#  Part B — 灵敏度分析（弹性法）
# ============================================================

def elasticity(p_name, T, d_p, Q, eps=0.02):
    """
    计算参数 p 对 t₉₅ 的弹性：S_p = ∂ln(Y)/∂ln(p)
    采用有限差分：对 ln(p) 加 ε 扰动
    """
    Y0 = t_95(T, d_p, Q)

    if p_name == 'T':
        T_new = np.exp(np.log(T + 273.15) + eps) - 273.15
        Y_new = t_95(T_new, d_p, Q)
    elif p_name == 'd_p':
        Y_new = t_95(T, d_p * np.exp(eps), Q)
    elif p_name == 'Q':
        Y_new = t_95(T, d_p, Q * np.exp(eps))
    else:
        raise ValueError(f"Unknown parameter: {p_name}")

    return (np.log(Y_new) - np.log(Y0)) / eps


def sensitivity_report(T=140, d_p=500, Q=12):
    """打印灵敏度分析报告"""
    sT  = elasticity('T',   T, d_p, Q)
    sD  = elasticity('d_p', T, d_p, Q)
    sQ  = elasticity('Q',   T, d_p, Q)
    print("=" * 60)
    print(f"灵敏度分析（T={T}℃, d_p={d_p}µm, Q={Q} L/min）")
    print(f"  温度弹性 S_T  = {sT:+.2f}  （阿伦尼乌斯指数主导）")
    print(f"  粒径弹性 S_d  = {sD:+.2f}  （内扩散平方关系）")
    print(f"  流量弹性 S_Q  = {sQ:+.2f}  （对流传质 0.6 次幂）")
    print(f"  结论：温度最敏感 → 但受品质约束；粒径受压降约束封顶")
    print("=" * 60)
    return sT, sD, sQ


# ============================================================
#  Part C — 稳态经济优化（约束 NLP 网格搜索）
# ============================================================

def optimize(X_target, dP_max=6.0, T_max=165, t_max=90):
    """
    网格搜索最优工况 (T*, d_p*, Q*)

    目标：min  J = 0.01·T + 0.05·Q + 0.08·t_batch + φ_penalty(T)
    约束：萃取率达标 | 压降 ≤ dP_max | 温度 ≤ T_max | 周期 ≤ t_max

    返回 dict 包含最优解
    """
    best = None
    for T in range(90, 176, 2):
        for d_p in range(250, 1025, 25):
            for Q in range(5, 25, 1):
                k = k_eff(T, d_p, Q)
                t_need = -np.log(1 - X_target / Xmax) / k

                if t_need > t_max:
                    continue
                if dP_ergun(d_p, Q) > dP_max:
                    continue
                if T > T_max:
                    continue

                # 品质惩罚：超过 150℃ 后每度加罚
                quality_penalty = 0.05 * (T - 150) if T > 150 else 0
                J = 0.01 * T + 0.05 * Q + 0.08 * t_need + quality_penalty

                if best is None or J < best['J']:
                    best = {'T': T, 'd_p': d_p, 'Q': Q, 'J': J,
                            't_need': t_need, 'dP': dP_ergun(d_p, Q)}

    return best


def optimize_report(X_targets=None):
    """打印多目标优化报告"""
    if X_targets is None:
        X_targets = [0.80, 0.85, 0.90, 0.95, 0.99]

    print("\n" + "=" * 60)
    print("稳态经济优化 — 网格搜索结果")
    print(f"{'X_target':>10} {'T*':>6} {'d_p*':>6} {'Q*':>5} {'J*':>7} {'t_need':>7} {'ΔP':>5}")
    print("-" * 60)
    for xt in X_targets:
        b = optimize(xt)
        if b:
            print(f"{xt:10.2f} {b['T']:>6}℃ {b['d_p']:>6}µm {b['Q']:>5}L/m "
                  f"{b['J']:>7.2f} {b['t_need']:>7.1f}m {b['dP']:>5.1f}b")
        else:
            print(f"{xt:10.2f}  无可行解")
    print("=" * 60)


# ============================================================
#  Part D — 可视化
# ============================================================

def plot_all(T=140, d_p=500, Q=12, save_path=None):
    """
    生成四合一图：
      (1) 萃取曲线 + 参数扰动对比
      (2) 灵敏度柱状图
      (3) 优化成本 J(T) 曲线 + 最优工作点
      (4) 粒径-压降权衡曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("速溶咖啡工业级萃取 · 建模、灵敏度与优化",
                 fontsize=15, fontweight='bold', y=0.98)

    # --- (1) 萃取曲线 ---
    ax = axes[0, 0]
    t = np.linspace(0, 60, 200)
    # 基准
    ax.plot(t, X_of_t(t, T, d_p, Q), 'k-', lw=2.5, label=f'基准 T={T}℃, dₚ={d_p}µm, Q={Q}')
    # 升温
    ax.plot(t, X_of_t(t, T + 30, d_p, Q), 'r--', lw=1.5, label=f'升温 T={T+30}℃')
    # 磨细
    ax.plot(t, X_of_t(t, T, d_p // 2, Q), 'g--', lw=1.5, label=f'磨细 dₚ={d_p//2}µm')
    # 达标线
    ax.axhline(y=0.95 * Xmax, color='gray', ls=':', alpha=0.6)
    ax.text(50, 0.96, 't₉₅ 达标线', fontsize=9, color='gray', ha='right')
    ax.set_xlabel('时间 t / min'); ax.set_ylabel('萃取率 X')
    ax.set_title('(a) 萃取动力学曲线', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # --- (2) 灵敏度 ---
    ax = axes[0, 1]
    sT, sD, sQ = sensitivity_report(T, d_p, Q)
    vals = [sT, sD, sQ]
    colors = ['#c0392b' if v < 0 else '#3f7d6e' for v in vals]
    bars = ax.bar(['温度 T', '粒径 dₚ', '流量 Q'], vals, color=colors, width=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3 * np.sign(v),
                f'{v:+.1f}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='#666', lw=1)
    ax.set_ylabel('弹性 S = ∂ln(t₉₅)/∂ln(p)')
    ax.set_title('(b) 参数灵敏度（弹性法）', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # --- (3) 优化成本曲线 ---
    ax = axes[1, 0]
    Xt = 0.95
    best = optimize(Xt)
    # 在最优粒径和流量下扫温度
    T_scan = np.arange(90, 176, 2)
    J_scan = []
    for Ti in T_scan:
        k = k_eff(Ti, best['d_p'], best['Q'])
        t_need = -np.log(1 - Xt / Xmax) / k
        if t_need <= 90 and dP_ergun(best['d_p'], best['Q']) <= 6 and Ti <= 165:
            penalty = 0.05 * (Ti - 150) if Ti > 150 else 0
            J_scan.append(0.01 * Ti + 0.05 * best['Q'] + 0.08 * t_need + penalty)
        else:
            J_scan.append(np.nan)
    ax.plot(T_scan, J_scan, 'brown', lw=2)
    ax.plot(best['T'], best['J'], '*', color='#d98a3d', markersize=18,
            label=f"最优 T*={best['T']}℃, J*={best['J']:.2f}")
    ax.set_xlabel('温度 T / ℃'); ax.set_ylabel('单位成本 J')
    ax.set_title(f'(c) 成本优化（X_target={Xt}）', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # --- (4) 粒径-压降权衡 ---
    ax = axes[1, 1]
    d_scan = np.arange(200, 1050, 25)
    dp_vals = [dP_ergun(d, Q) for d in d_scan]
    ax.plot(d_scan, dp_vals, 'teal', lw=2.5)
    ax.axhline(y=6, color='red', ls='--', alpha=0.6, label='ΔP_max=6 bar')
    ax.fill_between(d_scan, 0, 6, alpha=0.08, color='green')
    ax.fill_between(d_scan, 6, max(dp_vals) + 1, alpha=0.1, color='red')
    ax.set_xlabel('粒径 dₚ / µm'); ax.set_ylabel('压降 ΔP / bar')
    ax.set_title(f'(d) 粒径–压降权衡（Q={Q} L/min）', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至 {save_path}")
    plt.show()
    return fig


# ============================================================
#  交互演示入口
# ============================================================

def demo():
    """完整演示：打印报告 + 生成四合一图表"""
    print("\n" + "█" * 60)
    print("  速溶咖啡工业级萃取 · 化工三传一反模型")
    print("  课件第 4–6 章 Python 实现")
    print("█" * 60)

    T, d_p, Q = 140, 500, 12

    # 基准工况
    k = k_eff(T, d_p, Q)
    print(f"\n基准工况：T={T}℃  d_p={d_p}µm  Q={Q} L/min")
    print(f"  速率常数 k = {k:.4f} /min")
    print(f"  达标时间 t₉₅ = {t_95(T, d_p, Q):.1f} min")
    print(f"  30min 萃取率 = {X_of_t(30, T, d_p, Q) * 100:.1f}%")
    print(f"  压降 ΔP ≈ {dP_ergun(d_p, Q):.2f} bar")

    # 灵敏度
    sensitivity_report(T, d_p, Q)

    # 优化
    optimize_report([0.80, 0.85, 0.90, 0.95, 0.99])

    # 图表
    fig = plot_all(T, d_p, Q)
    return fig


# ============================================================
#  命令行入口
# ============================================================
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        fig = plot_all(save_path='/workspace/extraction_model_plots.png')
        demo()
    else:
        demo()
