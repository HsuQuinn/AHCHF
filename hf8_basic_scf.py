import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional


# ============================================================
# 0. 几何：三角晶格 + BZ + 高对称点路径
# ============================================================

def triangular_lattice(a: float = 1.0):
    """
    三角晶格：
      a1 = L (1/2, -√3/2)
      a2 = L (1/2,  √3/2)
      g1 = (2π/L) (1, -1/√3)
      g2 = (2π/L) (1,  1/√3)

    选 L 使原胞面积 A_uc = π a^2。
    """
    L = math.sqrt(2.0 * math.pi / 3.0) * a
    a1 = L * np.array([0.5, -math.sqrt(3.0) / 2.0])
    a2 = L * np.array([0.5,  math.sqrt(3.0) / 2.0])
    g1 = (2.0 * math.pi / L) * np.array([1.0, -1.0 / math.sqrt(3.0)])
    g2 = (2.0 * math.pi / L) * np.array([1.0,  1.0 / math.sqrt(3.0)])
    A_uc = abs(a1[0]*a2[1] - a1[1]*a2[0])
    return a1, a2, g1, g2, A_uc


def mp_grid_triangular(
    N1: int, N2: int,
    g1: np.ndarray, g2: np.ndarray,
    shift1: float = 0.0, shift2: float = 0.0
):
    """
    三角 BZ 里的 Monkhorst–Pack 网格 (N1 x N2)。
    返回:
      kpoints: (Nk,2)
      k_int:   (Nk,2) 对应整数坐标 (n1,n2)
    """
    kpoints = []
    k_int = []
    for n1 in range(N1):
        for n2 in range(N2):
            k1 = (n1 + shift1)/N1 - 0.5
            k2 = (n2 + shift2)/N2 - 0.5
            k = k1*g1 + k2*g2
            kpoints.append(k)
            k_int.append((n1, n2))
    return np.array(kpoints), np.array(k_int, dtype=int)


def high_sym_points(g1: np.ndarray, g2: np.ndarray):
    """
    三角 BZ 高对称点：
        K  = (g1 + g2)/3
        K' = (g2 - 2 g1)/3
        M  = g2/2
    """
    Gamma = np.array([0.0, 0.0])
    K  = (g1 + g2) / 3.0
    Kp = (-g2 - g1) / 3.0
    M  = -g2 / 2.0
    return K, Gamma, M, Kp


def build_path_KGMKp(
    g1: np.ndarray, g2: np.ndarray,
    n_per_seg: int = 400
):
    """
    连续路径 K–Γ–M–K'，按弧长参数化（只用于画图的横坐标）。
    """
    K, Gamma, M, Kp = high_sym_points(g1, g2)
    pts = [K, Gamma, M, Kp]

    ks = []
    s  = []
    acc = 0.0
    for seg in range(len(pts) - 1):
        a = pts[seg]
        b = pts[seg + 1]
        for j in range(n_per_seg):
            t = j / (n_per_seg - 1)
            k = (1.0 - t)*a + t*b
            if seg == 0 and j == 0:
                ks.append(k); s.append(0.0)
            else:
                dk = k - ks[-1]
                acc += np.linalg.norm(dk)
                ks.append(k); s.append(acc)
    ks = np.array(ks); s = np.array(s)

    L01 = np.linalg.norm(Gamma - K)
    L12 = np.linalg.norm(M - Gamma)
    L23 = np.linalg.norm(Kp - M)
    xticks = [0.0, L01, L01 + L12, L01 + L12 + L23]
    labels = ["K", "G", "M", r"K'"]
    return ks, s, xticks, labels


# ============================================================
# 1. λ-jellium 单带：spinor, overlap, 动能, Coulomb
# ============================================================

def spinor_lower_band(qvec: np.ndarray, lam: float) -> np.ndarray:
    """
    Eq.(3) 的下带 spinor：
      φ_q = [1, λ(q_x + i q_y)]^T / sqrt(1 + λ^2 q^2)
    """
    qx, qy = float(qvec[0]), float(qvec[1])
    q2 = qx*qx + qy*qy
    if q2 < 1e-16 or abs(lam) < 1e-16:
        return np.array([1.0+0j, 0.0+0j])
    denom = math.sqrt(1.0 + lam*lam*q2)
    return np.array([1.0, lam*(qx + 1j*qy)], dtype=np.complex128) / denom


def spinor_overlap(p1: np.ndarray, p2: np.ndarray, lam: float) -> complex:
    """〈u_{p1}|u_{p2}〉"""
    u1 = spinor_lower_band(p1, lam)
    u2 = spinor_lower_band(p2, lam)
    return np.vdot(u1, u2)


def kinetic_eps(kvec: np.ndarray, rs: float) -> float:
    """ε_k = |k|^2 / (2 r_s^2)"""
    '''or ε_k = |k|^2 / (r_s^2)'''
    k2 = float(kvec[0]**2 + kvec[1]**2)
    return k2 / (2.0*rs*rs)
    # return k2 / (rs*rs)


def coulomb_V(qvec: np.ndarray, rs: float, q_cut: float = 0.0) -> float:
    """
    2D jellium Coulomb：
      V(q) ∝ 1 / (r_s |q|)

    q=0 → 0；q_cut>0 用软截断。
    """
    q2 = float(qvec[0]**2 + qvec[1]**2)
    if q2 < 1e-6:
        return 0.0
    if q_cut > 0.0:
        return 4.0*math.pi/rs * (1.0/math.sqrt(q2 + q_cut*q_cut) - 1.0/q_cut)
    q = math.sqrt(q2)
    return 4.0*math.pi / (rs*q)


# ============================================================
# 2. G-shell + k+q index
# ============================================================

def momentum_shell(
    g1: np.ndarray, g2: np.ndarray,
    Ns: int
) -> List[Tuple[int, int, np.ndarray]]:
    """
    S = { G = m1 g1 + m2 g2 | |G| <= Ns * max(|g1|,|g2|) }，按 |G| 排序。
    返回 [(m1,m2,G), ...]
    """
    S = []
    norms = []
    gmax = Ns * max(np.linalg.norm(g1), np.linalg.norm(g2))
    for m1 in range(-Ns, Ns+1):
        for m2 in range(-Ns, Ns+1):
            G = m1*g1 + m2*g2
            n = np.linalg.norm(G)
            if n <= gmax + 1e-12:
                S.append((m1, m2, G))
                norms.append(n)
    norms = np.array(norms)
    order = np.argsort(norms)
    return [S[i] for i in order]


# ============================================================
# 3. HF 参数
# ============================================================

@dataclass
class HFParams:
    rs: float
    lam: float
    N1: int
    N2: int
    Nb: int = 7            # 默认 Nb=7，可改 13,19,37...
    filling: float = 1.0   # 每个 k 平均占据 filling 个态 (0~Nb)，总电子数 ≈ Nk*filling
    mix_alpha: float = 0.5  # 线性 mixing 步长
    Ns_shell: int = 4      # G-shell 截断
    q_cut: float = 0.0
    max_iter: int = 40
    conv_tol: float = 1e-5
    q_rep: int = 1         # q 网格扩展到 ±q_rep 个 BZ 副本


# ============================================================
# 4. 预计算单粒子 + form factor
# ============================================================

def build_geometry_and_kgrid(N1: int, N2: int):
    a1, a2, g1, g2, A_uc = triangular_lattice(a=1.0)
    kpoints, k_int = mp_grid_triangular(N1, N2, g1, g2)
    Nk = kpoints.shape[0]
    A_tot = Nk * A_uc
    return (a1, a2, g1, g2, A_uc, A_tot), kpoints, k_int


def build_G_shell(g1: np.ndarray, g2: np.ndarray,
                  Ns_shell: int, Nb: int):
    S_list = momentum_shell(g1, g2, Ns_shell)
    if Nb > len(S_list):
        raise ValueError(f"Nb={Nb} > |S|={len(S_list)} for Ns_shell={Ns_shell}")
    return S_list[:Nb]   # 前 Nb 个 G（包含 G=0）


def precompute_single_particle(rs: float, lam: float, params: HFParams):
    geom, kpoints, k_int = build_geometry_and_kgrid(params.N1, params.N2)
    a1, a2, g1, g2, A_uc, A_tot = geom
    Nk = kpoints.shape[0]

    S_used = build_G_shell(g1, g2, params.Ns_shell, params.Nb)
    G_list = [entry[2] for entry in S_used]             # (Nb,2)
    m_list = [(entry[0], entry[1]) for entry in S_used] # (m1,m2) list

    Nb = params.Nb
    h0 = np.zeros((Nk, Nb, Nb), dtype=np.complex128)
    for ik, k in enumerate(kpoints):
        for a, Ga in enumerate(G_list):
            p = k + Ga
            h0[ik, a, a] = kinetic_eps(p, rs)
    return geom, kpoints, k_int, np.array(G_list), np.array(m_list, dtype=int), h0


def build_extended_q_grid(
    N1: int, N2: int,
    g1: np.ndarray, g2: np.ndarray,
    q_rep: int = 1,
    q_norm_cut: Optional[float] = None,
):
    """
    构造扩展 q 网格：
      q = (m1/N1)*g1 + (m2/N2)*g2

    m1,m2 从多个 BZ 副本中取：
      m1 = n1 + s1*N1
      m2 = n2 + s2*N2
    其中 n1,n2=0..N1-1,  s1,s2=-q_rep..q_rep

    为避免重复，用 set 去重。可选按 |q|<=q_norm_cut 截断。
    返回:
      qpoints: (Nq,2) 实空间向量
      q_int:   (Nq,2) 对应整数 (m1,m2)
    """
    qpoints = []
    q_int = []
    seen = set()

    if q_norm_cut is None:
        base = max(np.linalg.norm(g1), np.linalg.norm(g2))
        q_norm_cut = (q_rep + 0.5) * base

    for s1 in range(-q_rep, q_rep + 1):
        for s2 in range(-q_rep, q_rep + 1):
            for n1 in range(N1):
                for n2 in range(N2):
                    m1 = n1 + s1 * N1
                    m2 = n2 + s2 * N2
                    key = (m1, m2)
                    if key in seen:
                        continue
                    qvec = (m1 / N1) * g1 + (m2 / N2) * g2
                    if np.linalg.norm(qvec) > q_norm_cut:
                        continue
                    seen.add(key)
                    q_int.append(key)
                    qpoints.append(qvec)

    qpoints = np.array(qpoints, dtype=float)
    q_int = np.array(q_int, dtype=int)
    print(f"[q-grid] Nq={len(qpoints)}, q_rep={q_rep}, q_norm_cut={q_norm_cut:.3f}")
    return qpoints, q_int


def precompute_form_factors(params: HFParams,
                            geom,
                            kpoints: np.ndarray,
                            k_int: np.ndarray,
                            G_list: np.ndarray,
                            m_list: np.ndarray):
    """
    与原版含义相同：预计算 Λ_g 和 Λ_q，
    但额外返回：
      - kplusq_index[iq,ik] = 对应 k+q 的 k 索引
      - Vg_arr[ig] = V(G_ig)
      - Vq_arr[iq] = V(q_iq)
    这些量在 HF 迭代中重复使用，避免重复计算。

    返回:
      G_list, Lambda_g, Lambda_q, qpoints, q_int,
      kplusq_index, Vg_arr, Vq_arr
    """
    rs, lam = params.rs, params.lam
    N1, N2 = params.N1, params.N2
    a1, a2, g1, g2, A_uc, A_tot = geom

    Nk = kpoints.shape[0]
    Nb = params.Nb

    # (m1,m2) -> band index
    m_to_index = {(int(m1), int(m2)): idx
                  for idx, (m1, m2) in enumerate(m_list)}

    Ng = len(G_list)

    # ---------- Λ_g(k) ----------
    Lambda_g = np.zeros((Ng, Nk, Nb, Nb), dtype=np.complex128)
    for ig in range(Ng):
        gvec = G_list[ig]
        m1g, m2g = m_list[ig]
        for ik, k in enumerate(kpoints):
            for a, (m1a, m2a) in enumerate(m_list):
                # G_b = G_a + g  ⇒ (m1b,m2b) = (m1a+m1g,m2a+m2g)
                key_b = (m1a + m1g, m2a + m2g)
                b = m_to_index.get(key_b, None)
                if b is None:
                    continue
                p_a = k + G_list[a]
                p_b = k + G_list[b]
                ov = spinor_overlap(p_a, p_b, lam)
                Lambda_g[ig, ik, a, b] = ov

    # ---------- 扩展 q 网格 ----------
    qpoints, q_int = build_extended_q_grid(N1, N2, g1, g2, q_rep=params.q_rep)
    Nq = qpoints.shape[0]

    Lambda_q = np.zeros((Nq, Nk, Nb, Nb), dtype=np.complex128)

    for iq, qvec in enumerate(qpoints):
        m1q, m2q = q_int[iq]

        for ik, k in enumerate(kpoints):
            n1k, n2k = k_int[ik]

            # k + q 的“总整数坐标”
            n1_tot = n1k + m1q
            n2_tot = n2k + m2q

            # [k+q] 对应的 torus 坐标
            n1p = n1_tot % N1
            n2p = n2_tot % N2

            # alias 倒格矢系数 ℓ1,ℓ2
            ell1 = (n1_tot - n1p) // N1
            ell2 = (n2_tot - n2p) // N2

            for b, (m1b, m2b) in enumerate(m_list):
                # G_a = G_b - g_ℓ ⇒ m_a = m_b - (ell1,ell2)
                key_a = (m1b - ell1, m2b - ell2)
                a = m_to_index.get(key_a, None)
                if a is None:
                    continue

                p_a = k + G_list[a]            # p_a = k + G_a
                # [k+q] + G_b = p_a + qvec
                ov = spinor_overlap(p_a, p_a + qvec, lam)
                Lambda_q[iq, ik, a, b] = ov

    # ---------- 预计算 k+q 的索引表 kplusq_index[iq,ik] ----------
    kplusq_index = np.zeros((Nq, Nk), dtype=np.int32)
    for iq in range(Nq):
        m1q, m2q = q_int[iq]
        for ik in range(Nk):
            n1k, n2k = k_int[ik]
            n1_tot = n1k + m1q
            n2_tot = n2k + m2q
            n1p = n1_tot % N1
            n2p = n2_tot % N2
            kplusq_index[iq, ik] = n1p * N2 + n2p

    # ---------- 预计算 Coulomb 势 V(G) 和 V(q) ----------
    Vg_arr = np.array([coulomb_V(G_list[ig], rs, q_cut=params.q_cut) for ig in range(Ng)],
                      dtype=float)
    Vq_arr = np.array([coulomb_V(qpoints[iq], rs, q_cut=params.q_cut) for iq in range(Nq)],
                      dtype=float)

    return G_list, Lambda_g, Lambda_q, qpoints, q_int, kplusq_index, Vg_arr, Vq_arr

def self_consistent_HF(params: HFParams):
    rs, lam = params.rs, params.lam
    Nb = params.Nb

    if not (0.0 <= params.filling <= Nb):
        raise ValueError(f"filling must be in [0, Nb], got {params.filling}")

    # ---------- 预计算单粒子 ----------
    geom, kpoints, k_int, G_list, m_list, h0 = precompute_single_particle(rs, lam, params)
    a1, a2, g1, g2, A_uc, A_tot = geom
    Nk = kpoints.shape[0]

    print(f"[geom] N1={params.N1}, N2={params.N2}, Nk={Nk}, Nb={Nb}, A_tot={A_tot:.4f}")

    # ---------- 预计算 form factors + k+q 索引 + 势 ----------
    (G_list,
     Lambda_g,
     Lambda_q,
     qpoints,
     q_int,
     kplusq_index,
     Vg_arr,
     Vq_arr) = precompute_form_factors(
        params, geom, kpoints, k_int, G_list, m_list
    )
    Nq = qpoints.shape[0]

    # 总电子数：Ne_tot ≈ Nk * filling
    Ne_tot = int(round(params.filling * Nk))
    Ne_tot = max(0, min(Ne_tot, Nk * Nb))
    print(f"[filling] filling={params.filling:.4f}, Ne_tot={Ne_tot}, Nk={Nk}, Nb={Nb}")

    # ---------- 初始 P：均匀 rank-1 投影 ----------
    phi = np.ones(Nb, dtype=np.complex128) / np.sqrt(Nb)
    P0 = np.outer(phi, phi.conj())
    P_old = np.zeros((Nk, Nb, Nb), dtype=np.complex128)
    for ik in range(Nk):
        P_old[ik] = P0.copy()

    print("[init] Tr(P[k]) (k=0..3) =", [np.trace(P_old[ik]).real for ik in range(min(4, Nk))])

    # ---------- 给定 P 计算 hH, hF, F, E[P] ----------
    def build_HF_operators(Pmat: np.ndarray):
        # Hartree
        rho_g = np.einsum("kab,ikba->i", Pmat, Lambda_g.conj(), optimize=True)
        C = Vg_arr * rho_g
        hH = np.einsum("i,ikab->kab", C, Lambda_g, optimize=True) / A_tot

        # Fock
        hF = np.zeros_like(Pmat)
        for iq in range(Nq):
            Vq = Vq_arr[iq]
            if abs(Vq) < 1e-16:
                continue
            L_q = Lambda_q[iq]              # (Nk,Nb,Nb)
            P_plus = Pmat[kplusq_index[iq]] # (Nk,Nb,Nb)
            S = np.einsum("kab,kbc->kac", L_q, P_plus, optimize=True)
            hF_q = np.einsum("kac,kbc->kab", S, L_q.conj(), optimize=True)
            hF += -Vq * hF_q / A_tot

        F = h0 + hH + hF
        K = 2.0 * h0 + hH + hF   # 用于能量：E = 1/2 Tr(P K)
        E = 0.5 * np.einsum("kab,kab->", Pmat, K).real
        return hH, hF, F, E

    def commutator_norm(Pmat: np.ndarray, Fmat: np.ndarray) -> float:
        comm_max = 0.0
        for ik in range(Nk):
            comm = Pmat[ik] @ Fmat[ik] - Fmat[ik] @ Pmat[ik]
            nrm = np.linalg.norm(comm)
            if nrm > comm_max:
                comm_max = nrm
        return comm_max

    # ---------- 初始 F, E ----------
    _, _, F_old, E_old = build_HF_operators(P_old)
    print(f"[HF] iter  0: E = {E_old:.8f}, max||[P,F]|| = {commutator_norm(P_old,F_old):.3e}")

    for it in range(1, params.max_iter + 1):
        # ===== 1) 用 F_old 求“完更新”的投影 P_scf（全局填充） =====
        eigvals_all = np.zeros((Nk, Nb))
        eigvecs_all = np.zeros((Nk, Nb, Nb), dtype=np.complex128)

        for ik in range(Nk):
            evals, evecs = np.linalg.eigh(F_old[ik])
            idx = np.argsort(evals.real)
            evals = evals[idx].real
            evecs = evecs[:, idx]
            eigvals_all[ik] = evals
            eigvecs_all[ik] = evecs

        # 全局排序 (ik, band) 上的本征值
        flat = [(eigvals_all[ik, a], ik, a)
                for ik in range(Nk) for a in range(Nb)]
        flat.sort(key=lambda x: x[0])

        occ_flags = np.zeros((Nk, Nb), dtype=bool)
        for n in range(Ne_tot):
            _, ik_occ, a_occ = flat[n]
            occ_flags[ik_occ, a_occ] = True

        P_scf = np.zeros_like(P_old)
        for ik in range(Nk):
            occ_idx = np.where(occ_flags[ik])[0]
            if len(occ_idx) == 0:
                continue
            vecs = eigvecs_all[ik][:, occ_idx]
            P_scf[ik] = vecs @ vecs.conj().T

        # 在真正的投影 P_scf 上算一次“物理”能量 & 公度子
        _, _, F_scf, E_scf = build_HF_operators(P_scf)
        comm_scf = commutator_norm(P_scf, F_scf)

        # ===== 2) 沿 P_old -> P_scf 做 backtracking line search，强制 E 不增 =====
        D = P_scf - P_old
        alpha = params.mix_alpha      # 初始步长，用你传进来的 mix_alpha
        alpha_min = 1e-4              # 最小步长
        best_E = E_old
        best_alpha = 0.0
        F_best = F_old

        while alpha >= alpha_min:
            P_trial = P_old + alpha * D
            _, _, F_trial, E_trial = build_HF_operators(P_trial)
            if E_trial <= best_E:
                best_E = E_trial
                best_alpha = alpha
                F_best = F_trial
                break
            alpha *= 0.5  # backtracking

        if best_alpha == 0.0:
            # 找不到下降方向，认为到了极小附近（或者 saddle），退出
            print(f"[HF] iter {it:3d}: no descent direction found "
                  f"(E_old={E_old:.8f}, E_scf={E_scf:.8f}). Stop.")
            P_new, F_new, E_new = P_old, F_old, E_old
            break
        else:
            P_new = P_old + best_alpha * D
            F_new = F_best
            E_new = best_E

        # 注意这里不再除 Nk，直接看实际 max 变化
        DeltaP = np.max(np.abs(P_new - P_old))

        print(f"[HF] iter {it:3d}: "
              f"E = {E_new:.8f} (E_old={E_old:.8f}, E_scf={E_scf:.8f}, α={best_alpha:.3e}), "
              f"|ΔP|_max = {DeltaP:.3e}, "
              f"||[P_scf,F_scf]||_max = {comm_scf:.3e}")

        # 收敛判据：P 变化小 + 公度子小
        if DeltaP < params.conv_tol:
            print("[HF] Converged by ΔP & commutator.")
            P_old, F_old, E_old = P_new, F_new, E_new
            break

        P_old, F_old, E_old = P_new, F_new, E_new

    # ---------- 最终本征值 ----------
    E_k = np.zeros((Nk, Nb))
    for ik in range(Nk):
        evals, _ = np.linalg.eigh(F_old[ik])
        E_k[ik] = evals.real

    # 检查各 k 的占据数分布，确认是“全局填充”的金属/部分填充行为
    occ_per_k = np.array([np.trace(P_old[ik]).real for ik in range(Nk)])
    print("Tr(P_k) min / max / avg = ",
          occ_per_k.min(), occ_per_k.max(), occ_per_k.mean())
    # 收敛之后

    rho_g = np.einsum("kab,ikba->i", P_old, Lambda_g.conj(), optimize=True)
    print("rho_g[0] = ", rho_g[0])
    print("max |rho_g(g≠0)| = ", np.max(np.abs(rho_g[1:])))


    return kpoints, P_old, E_k, geom, F_old


# ============================================================
# 6. 用收敛的 HF Hamiltonian 沿 MP 网格上的 K–Γ–M–K' 路径画 band
# ============================================================

def straight_line_int(n1a, n2a, n1b, n2b):
    """
    在整数格点坐标 (n1,n2) 上，取端点 (n1a,n2a) 和 (n1b,n2b) 之间
    那条“真正的直线”上的所有格点：
        (n1, n2) = (n1a + m*Δn1/g, n2a + m*Δn2/g),  m=0..g
    其中 g = gcd(|Δn1|, |Δn2|)。

    返回 [(n1,n2), ...]，包含起点和终点。
    """
    dn1 = n1b - n1a
    dn2 = n2b - n2a
    if dn1 == 0 and dn2 == 0:
        return [(n1a, n2a)]

    g_val = math.gcd(abs(dn1), abs(dn2))
    step1 = dn1 // g_val
    step2 = dn2 // g_val

    path = []
    for m in range(g_val + 1):
        n1 = n1a + m * step1
        n2 = n2a + m * step2
        path.append((n1, n2))
    print(n1a, n2a, n1b, n2b, path)
    return path


def build_kpath_on_MP_grid(kpoints: np.ndarray,
                            geom,
                            N1: int, N2: int):
    """
    在给定 MP 网格上构造一条离散的 K–Γ–M–K' 路径
    """
    a1, a2, g1, g2, A_uc, A_tot = geom

    K, Gamma, M, Kp = high_sym_points(g1, g2)
    hs_vecs = [K, Gamma, M, Kp]

    hs_int = []
    for v in hs_vecs:
        d2 = np.sum((kpoints - v)**2, axis=1)
        ik = int(np.argmin(d2))
        n1 = ik // N2
        n2 = ik % N2
        hs_int.append((n1, n2))

    path_int = []
    s = [0.0]
    xticks = [0.0]

    def k_from_int(n1, n2):
        return kpoints[n1 * N2 + n2]
    
    for seg in range(3):  # 三段：K-Γ, Γ-M, M-K'
        n1a, n2a = hs_int[seg]
        n1b, n2b = hs_int[seg + 1]
        seg_path = straight_line_int(n1a, n2a, n1b, n2b)
        if seg > 0:
            seg_path = seg_path[1:]

        for (n1, n2) in seg_path:
            if not path_int:
                path_int.append((n1, n2))
                continue
            n1_prev, n2_prev = path_int[-1]
            k_prev = k_from_int(n1_prev, n2_prev)
            k_curr = k_from_int(n1, n2)
            ds = np.linalg.norm(k_curr - k_prev)
            s.append(s[-1] + ds)
            path_int.append((n1, n2))

        xticks.append(s[-1])

    indices = np.array([n1 * N2 + n2 for (n1, n2) in path_int], dtype=int)
    s = np.array(s)
    labels = ["K", "G", "M", r"K'"]

    return indices, s, xticks, labels


def plot_bands_from_F_on_MP(F: np.ndarray,
                            kpoints: np.ndarray,
                            geom,
                            N1: int, N2: int,
                            Nb: int):
    """
    F:   (Nk,Nb,Nb) 收敛 HF Hamiltonian
    沿 MP 网格上的离散 K–Γ–M–K' 路径画 Nb 条 band
    """
    Nk = kpoints.shape[0]

    eigvals = np.zeros((Nk, Nb))
    for ik in range(Nk):
        evals, _ = np.linalg.eigh(F[ik])
        eigvals[ik] = evals.real

    indices, s, xticks, labels = build_kpath_on_MP_grid(kpoints, geom, N1, N2)

    E_path = eigvals[indices, :]

    plt.figure(figsize=(4, 8))
    for a in range(Nb):
        plt.plot(s, E_path[:, a], lw=2.0, color="r")

    for xt in xticks:
        plt.axvline(xt, color="k", lw=0.5)

    plt.xticks(xticks, labels)
    plt.xlabel(r"$k$ along K–G–M–K'")
    plt.ylabel(r"$E_{\mathrm{HF}}(k)$")
    plt.title(f"HF bands (Nb={Nb}) along K–Γ–M–K' (on MP grid)")
    plt.tight_layout()
    plt.savefig("rs25_lambda0_Nq2_Ns5_ver2.png")
    plt.show()


# ============================================================
# 7. main 示例
# ============================================================

if __name__ == "__main__":
    rs  = 21
    lam = 0.0
    N1 = N2 = 18     # k 网格（最好 6 的倍数）
    Nb = 7           # 你可以改成 13, 19, 37 ...
    filling = 1.0    # spinless：平均每个 k 占据 1 个态 → Ne_tot ≈ Nk

    params = HFParams(
        rs=rs,
        lam=lam,
        N1=N1,
        N2=N2,
        Nb=Nb,
        filling=filling,
        mix_alpha=0.5,   # 建议 0.2~0.3，若还不稳再减小
        Ns_shell=5,
        q_cut=0.0,
        max_iter=50,
        conv_tol=2e-4,
        q_rep=2,
    )

    kpoints, P, E_k, geom, F_conv = self_consistent_HF(params)
    print("HF done. E_k shape =", E_k.shape,
          "min/max =", E_k.min(), E_k.max())
    

    plot_bands_from_F_on_MP(F_conv, kpoints, geom, N1, N2, Nb)
