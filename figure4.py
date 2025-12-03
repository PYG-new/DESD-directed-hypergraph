"""
Implementation of Dirac-Equation Synchronization Dynamics (DESD)
on directed hypergraphs used in the manuscript.

The code includes:
    (i)   Hypergraph Stochastic Block Model (hSBM) generator,
    (ii)  Construction of the Dirac operator H for hypergraphs,
    (iii) Selection of isolated Dirac eigenmodes,
    (iv)  Dirac synchronization dynamics integrated with RK4,
"""
from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix,bmat,eye
import scipy
from scipy.io import savemat, loadmat

# ================================================================
# Runge–Kutta 4 for integrating Dirac synchronization dynamics
# ================================================================
def Runge_Kutta(Ham_E,sigma,omega,Psi,dt):
    """
    Compute Ψ_dot using RK4 for the Dirac synchronization model.
    Dirac model:
        Ψ̇ = ω − σ * H^T sin(H Ψ)
    Returns
    -------
        Psi_dot : np.array
            Time derivative of Ψ
        """
    def Dirac_function(omega, Psi, Ham_E, sigma):
        delta_Psi = omega - sigma * Ham_E.T @ np.sin(Ham_E @ Psi)
        return delta_Psi
    k1 = Dirac_function(omega, Psi, Ham_E, sigma)
    k2 = Dirac_function(omega, Psi + 0.5 * dt * k1, Ham_E, sigma)
    k3 = Dirac_function(omega, Psi + 0.5 * dt * k2, Ham_E, sigma)
    k4 = Dirac_function(omega, Psi + dt * k3, Ham_E, sigma)

    Psi_dot = (k1+2*k2+2*k3+k4)/6

    return Psi_dot
# ============================================================
# 2. Isolated eigenvalue selection
# ============================================================
def Calculate_IsolateValue(eigValue,eigVector,m,C=4,text='model1',label='pos'):
    """
        Identify an isolated eigenvalue and its eigenvector from the Dirac spectrum.
    """
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigValue)
    sorted_Vals = np.real(eigValue[sorted_indices])
    sorted_Vecs = np.real(eigVector[:, sorted_indices])
    # --------------------------------------------------------
    # Model 1: direct threshold around ±m
    # --------------------------------------------------------
    if text == 'model1':
        # Indices of eigenvalues above +m and below -m
        pos_idx = np.where(sorted_Vals > m + 0.01)[0]
        neg_idx = np.where(sorted_Vals < -(m + 0.01))[0]
        # Positive isolated eigenvalue (take `id`-th one above gap)
        # position
        pos_value = sorted_Vals[pos_idx[0]]
        pos_vector = sorted_Vecs[:, pos_idx[0]]
        # negative
        neg_value = sorted_Vals[neg_idx[-1]]
        neg_vector = sorted_Vecs[:,neg_idx[-1]]
        if label=='pos':
            return pos_value, pos_vector, sorted_Vals, sorted_Vecs
        else:
            return neg_value, neg_vector, sorted_Vals, sorted_Vecs
    # --------------------------------------------------------
    # Model 2: choose eigenvalue with largest local spectral gap
    # separately on the positive and negative side.
    # --------------------------------------------------------
    elif text=='model2':
        n = len(sorted_Vals)
        index_poses = np.where(sorted_Vals > m + 0.01)[0][:C]
        index_neges = np.where(sorted_Vals < -(m + 0.01))[0][-1*C:]
        index_pos = index_poses[0]
        index_neg = index_neges[0]
        err_pos = 0
        err_neg = 0
        if label == 'pos':
            # scan positive candidates and pick the one with the largest min(left_gap, right_gap)
            for index in index_poses:
                evalue = sorted_Vals[index]
                if index == 0:
                    delta_left = 0
                else:
                    delta_left = evalue - sorted_Vals[index - 1]
                if index == n - 1:
                    delta_right = 0
                else:
                    delta_right = sorted_Vals[index + 1] - evalue
                if delta_left > err_pos and delta_right > err_pos and evalue > m+0.0001 :
                    index_pos = index
                    err_pos = min(delta_left,delta_right)
            # fallback to the first eigenvalue above m if nothing improved
            index_pos = np.where(sorted_Vals > m + 0.01)[0][0] if index_pos==0 else index_pos
            isolated_eigenvalues = np.array([sorted_Vals[index_pos]])
            isolated_eigenvector = np.array([sorted_Vecs[:,index_pos]])
            pos_value = isolated_eigenvalues[isolated_eigenvalues>(m+0.01)][0]
            pos_vector = isolated_eigenvector[isolated_eigenvalues > (m + 0.01)][0]
            return pos_value,pos_vector,sorted_Vals,sorted_Vecs
        else:
            # scan negative candidates
            for index in index_neges:
                evalue = sorted_Vals[index]
                if index == 0:
                    delta_left = 0
                else:
                    delta_left = evalue - sorted_Vals[index - 1]
                if index == n - 1:
                    delta_right = 0
                else:
                    delta_right = sorted_Vals[index + 1] - evalue
                if delta_left > err_neg and delta_right > err_neg and evalue < -(m + 0.0001):
                    index_neg = index
                    err_neg = min(delta_left, delta_right)
            # fallback to the last eigenvalue below -m
            index_neg = np.where(sorted_Vals < -(m + 0.01))[0][-1] if index_neg == 0 else index_neg
            isolated_eigenvalues = np.array([sorted_Vals[index_neg]])
            isolated_eigenvector = np.array([sorted_Vecs[:, index_neg]])
            neg_value = isolated_eigenvalues[isolated_eigenvalues < -(m + 0.01)][0]
            neg_vector = isolated_eigenvector[isolated_eigenvalues < -(m + 0.01)][0]

            return neg_value, neg_vector, sorted_Vals, sorted_Vecs
    else:
        raise Exception("Something went wrong")

# ============================================================
# 2. Dirac-Equation Synchronization Dynamics (DESD)
# ============================================================
def DESD_Dynamic_function(B,Psi0,sigma,m=1,C=4,T_max =20,step_t=0.01,label='pos'):
    """
    Compute Dirac-Equation Synchronization Dynamics (DESD) on a directed hypergraph.

    Steps:
        (1) Build Dirac operator H from incidence matrix B.
        (2) Identify isolated eigenmode ±E.
        (3) Construct natural frequencies ω in the Dirac basis.
        (4) Integrate Ψ̇ = ω − σ Hᵀ sin(H Ψ) using RK4.
    """
    N, M = B.shape  # #nodes, #hyperedges
    total = N + M  # dimension of topological signals (nodes + hyperedges)
    num_steps = int(T_max / step_t)

    # Preallocate containers (not filled in this version)
    Psi_T = np.zeros((total, num_steps + 1))
    Psi_dot_T = np.zeros((total, num_steps))
    D_Psi_T = np.zeros((total, num_steps + 1))

    # --------------------------------------------------------
    # Build Dirac-like operator H for the hypergraph
    # --------------------------------------------------------
    # D: incidence between node- and hyperedge-signals
    D = bmat([[csr_matrix((N, N)), B],
              [B.T, csr_matrix((M, M))]], format='csr')
    gamma = bmat([[eye(N, format='csr'), csr_matrix((N, M))],
                  [csr_matrix((M, N)), -eye(M, format='csr')]], format='csr')
    H = D + m * gamma

    # Full eigendecomposition of H
    eig_value, eig_vector = np.linalg.eigh(H.toarray())

    # Extract isolated eigenvalue and eigenvector
    E,Vector,eigValue,eigVector = Calculate_IsolateValue(eig_value,eig_vector,m,C-1,text='model2',label=label)
    pos = np.where(eigValue == E)
    print(pos)
    Psi = Psi0.copy()
    if label == 'pos':
        plot_simple_dirac_spectrum(eigValue, m)
    # Generate random natural frequencies for nodes and hyperedges
    Omega0, tau0 = 0, 1
    Omega1, tau1 = 0, 1
    w = Omega0 + np.random.randn(N, 1) * tau0
    wedge = Omega1 + np.random.randn(M, 1) * tau1
    # Change basis to eigenbasis of H
    Inv_eigvector = np.linalg.inv(eigVector)

    Omega = np.matmul(np.vstack([w, wedge]).T, Inv_eigvector).T

    pOmega = np.matmul(Inv_eigvector, Omega)
    # strengthen isolated mode
    pOmega[pos[0][0]] = 1
    alpha = 0.1
    for k in range(1,C):
        pOmega[pos[0][0] - k] = pOmega[pos[0][0] - k]*alpha
    for k in range(1,C):
        pOmega[pos[0][0] + k] = pOmega[pos[0][0] + k] * alpha

    omega = csr_matrix(eigVector @ pOmega)
    Ham_E = H - E*eye(total)
    # ------------------------------
    # Time integration
    # ------------------------------
    for i in range(num_steps):
        Psi_dot = Runge_Kutta(Ham_E,sigma,omega,Psi,step_t)

        Psi += Psi_dot*step_t

        Psi = csr_matrix(Psi)
        Psi_dot_T[:,i] = Psi_dot.toarray().squeeze()
        Psi_T[:,i+1] = Psi.toarray().squeeze()
        D_Psi_T[:,i] = (Psi_dot*step_t).toarray().squeeze()

        print(f'epoch:{i}')
    return Psi_T,Psi_dot_T,Vector,omega.toarray().T,E,eigVector,eigValue,Ham_E
# ========================================================================
# 4. Plotting: Dirac spectrum
# ========================================================================
def plot_simple_dirac_spectrum(sorted_eig_vals, m=1):
    """
    Plot empirical Dirac eigenvalue density ρ_c(E)
    and highlight the mass gap [−m, m].
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    neglam = sorted_eig_vals[sorted_eig_vals < 0]
    nneglam = sorted_eig_vals[sorted_eig_vals >= 0]
    dnneglam = np.flip(nneglam)
    rho_neg = np.arange(1, len(neglam) + 1) / len(sorted_eig_vals)
    rho_nneg = np.arange(1, len(nneglam) + 1) / len(sorted_eig_vals)

    fig, ax_main = plt.subplots(figsize=(3.5, 2.16))
    ax_main.set_title('Eigenvalue density spectrum',fontsize=12)
    ax_main.scatter(dnneglam, rho_nneg, facecolors='none', color='blue',s=15)
    ax_main.scatter(neglam, rho_neg, facecolors='none', color='blue',s=15)
    ax_main.add_patch(Rectangle((np.min(nneglam)-0.05,-0.01), np.max(nneglam)-np.min(nneglam)+0.1,np.max(rho_nneg)+0.02,fill=False,edgecolor='red', linestyle='--', linewidth=2))

    ax_main.set_xlabel('$E$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax_main.set_ylabel(r'$\rho_c(E)$', fontdict={'fontsize': 12, 'fontweight': 'bold'})

    ax_inset = inset_axes(ax_main, width="35%", height="35%", loc='upper right')
    ax_inset.scatter(dnneglam, rho_nneg,facecolors='none', color='red',s=15)
    ax_inset.tick_params(axis='x', labelsize=10)
    ax_inset.tick_params(axis='y', labelsize=10)

    plt.grid(False)
    plt.show()
# ========================================================================
# 5. Random directed hypergraph SBM generator
# ========================================================================
def hypergraph_SBM(V, C, K, c1=15, c2=1,seed=33):
    """
        Generate a directed hypergraph with community (SBM-like) structure.
        We construct:
        - Intra-community hyperedges: within each community.
        - Inter-community hyperedges: between different communities.
        Incidence matrix B is normalized such that:
            B[i, e] = -1 / |tail(e)|  if node i in tail of e
            B[i, e] =  1 / |head(e)|  if node i in head of e
    """
    N = V * C   # total number of nodes
    hyper_edges = []
    k = K[0]    # initial hyperedge size
    # --------------------------------------------------------
    # Intra-community hyperedges
    # --------------------------------------------------------
    for i in range(C):
        k = K[i]  # hyperedge size in community i
        rng = np.random.default_rng(seed + i * 10)
        nodes = np.arange(i * V, (i + 1) * V)  # node indices in community i

        # Random weights (not used in the final construction, but kept for legacy)
        weights = rng.random(len(nodes))
        weights /= weights.sum()
        # Number of intra-community hyperedges
        H_intra = int(round(c1 * V / k))

        for _ in range(H_intra):
            # sample k nodes from this community
            idx = rng.choice(range(V), k, replace=False)

            nodes_h = nodes[idx]
            # randomly split them into tail & head
            tail_idx = rng.choice(range(len(nodes_h)),np.random.randint(1,len(nodes_h)),replace=False)
            hyper_edge = {
                'tail': list(nodes_h[tail_idx]),
                'head': list(np.delete(nodes_h, tail_idx))
            }
            hyper_edges.append(hyper_edge)
    # --------------------------------------------------------
    # Inter-community hyperedges
    # --------------------------------------------------------
    k = 70
    Inter_num = len(hyper_edges)
    Inter_edges = []
    Inter_edges.append(Inter_num)   # starting index for inter-community hyperedges
    dict_Inter_edges = {}
    num = 0  # index for inter-blocks
    for i in range(C):
        rng_i = np.random.default_rng(seed+i * 10)
        for j in range(i+1, C):
            if i ==j:
                continue
            # node index ranges for communities i and j
            rng_j = np.random.default_rng(seed+j * 10)
            nodes_i = np.arange(i*V, (i+1)*V)
            nodes_j = np.arange(j*V, (j+1)*V)

            # number of inter-community hyperedges between (i,j)
            H_inter = int(round(c2*V/k))
            Inter_edges.append(Inter_edges[-1]+H_inter)
            dict_Inter_edges[num] = [i,j]
            num+=1
            # half hyperedges oriented i → j
            for ij in range(int(H_inter/2)):
                k1 = rng_i.integers(1,k//2,size=1)[0]
                k1 = np.random.randint(1,k)
                k2 = k - k1
                idx_i = rng_i.choice(V, min(k1, V), replace=False)
                idx_j = rng_j.choice(V, min(k2, V), replace=False)
                nodes_h = np.concatenate([nodes_i[idx_i], nodes_j[idx_j]])
                tail_idx = np.random.choice(range(len(nodes_h)), np.random.randint(1, len(nodes_h)), replace=False)

                hyper_edge = {
                    'tail': list(nodes_i[idx_i]),
                    'head': list(nodes_j[idx_j])
                }
                hyper_edges.append(hyper_edge)
            for ij in range(H_inter-int(H_inter/2)):
                k1 = rng_i.integers(1,k//2,size=1)[0]
                k1 = np.random.randint(1, k)
                k2 = k - k1
                idx_i = rng_i.choice(V, min(k1, V), replace=False)
                idx_j = rng_j.choice(V, min(k2, V), replace=False)
                nodes_h = np.concatenate([nodes_i[idx_i], nodes_j[idx_j]])
                tail_idx = np.random.choice(range(len(nodes_h)), np.random.randint(1, len(nodes_h)), replace=False)
                hyper_edge = {
                    'tail': list(nodes_j[idx_j]),
                    'head': list(nodes_i[idx_i])
                }
                hyper_edges.append(hyper_edge)

    # --------------------------------------------------------
    # Build incidence matrix B
    # --------------------------------------------------------
    L = len(hyper_edges)
    B = np.zeros((N, L))
    for l, hyper_edge in enumerate(hyper_edges):
        tail_len = len(hyper_edge['tail'])
        head_len = len(hyper_edge['head'])
        for t in hyper_edge['tail']:
            B[t, l] = -1 / tail_len
        for h in hyper_edge['head']:
            B[h, l] = 1 / head_len
    # Check connectivity on the bipartite (node-edge) graph
    D = np.block([[np.zeros((N,N)),B],[B.T,np.zeros((L,L))]])
    D = csr_matrix(D)
    G = nx.from_scipy_sparse_array(D)
    flag = 1 if nx.is_connected(G) else 0
    B = csr_matrix(B)
    return hyper_edges, B, flag,Inter_edges,dict_Inter_edges

if __name__=='__main__':
    seed=84
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Hypergraph and dynamics parameters
    # ------------------------------------------------------------------
    N = 50      # nodes per community
    C = 4       # number of communities
    dt = 0.001  # time step
    T_max = 5   # horizon
    m = 1       # mass term in Dirac operator
    sigma =50   # coupling strength (used in full dynamic version)
    # hyperedge size per community (here all =3, but length=10 unused)
    k = list(np.ones(4,dtype=int)*3)

    c = 25
    # --------------------------------------------------------
    # Generate hypergraph SBM instance
    # --------------------------------------------------------
    hyper_edges, B, flag,Inter_edges,dict_Inter_edges = hypergraph_SBM(N,C,k,c1=c,c2=c,seed=seed)
    nrun = 0
    while flag==0 and nrun<20:
        hyper_edges, B, flag,Inter_edges,dict_Inter_edges = hypergraph_SBM(N,C,k,c1=c,c2=c,seed=seed)
        nrun+=1

    num_edge = len(hyper_edges)
    # ------------------------------------------------------------------
    # Random initial phases for both nodes and hyperedges
    # Ψ(0) ∈ ℝ^{N·C + M}, here drawn from a normal distribution scaled by π
    # ------------------------------------------------------------------
    Psi0 = np.pi*np.random.randn(N*C+num_edge,1)
    Psi0 = csr_matrix(Psi0)
    color_list = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#A6CEE3', '#CC79A7']
    time = np.linspace(0, stop=T_max, num=int(T_max / dt))

    # ------------------------------------------------------------------
    # DESD driven by a positive isolated Dirac eigestate(Nodes)
    # ------------------------------------------------------------------
    Psi_T, Psi_dot_T, Psi_eigenstate, omega, E,eigen_matrix,eigen_Value,H_ham = DESD_Dynamic_function(B, Psi0, sigma, m=m, T_max=T_max, step_t=dt,label='pos')
    label = ['A', 'B', 'C', 'D']
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch

    # ------------------------------------------------------------------
    # Figure with 2 × 3 subpanels:
    #   Row 1: node-level dynamics and eigenstate projection
    #   Row 2: hyperedge-level dynamics and eigenstate projection
    # ------------------------------------------------------------------
    fig,axis = plt.subplots(2,3,figsize=(7,3.8))
    sub_fig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)','(g)']
    fig.tight_layout()
    for label, ax in zip(sub_fig_labels, axis.flat):
        pos = ax.get_position()
        fig.text(
            pos.x0-0.01,
            pos.y1+0.01,
            label,
            fontsize=12,
            va='bottom',
            ha='right'
        )
    for i in range(C):
        for j in range(N):
            theta_dot = Psi_dot_T[(N * i)+j,:].ravel()
            axis[0,0].plot(time, theta_dot, color=color_list[i],linewidth=1.0)
            axis[0, 1].plot(time[-1000:], theta_dot[-1000:],linewidth=1.0, color=color_list[i])

    axis[0,0].set_xlabel(r'$t$', fontsize=12)
    axis[0,0].set_ylabel(r'$\dot{\theta}_{nodes}$', fontsize=12)
    axis[0, 1].set_xlabel(r'$t$', fontsize=12)
    axis[0, 1].set_ylabel(r'$\dot{\theta}_{nodes}$', fontsize=12)
    axis[0, 0].set_ylim(-400, 400)
    axis[0, 1].set_ylim(-0.15, 0.15)
    axis[0,0].grid(False)
    axis[0,0].add_patch(
        Rectangle(( T_max-1,-7),1.01,15,
                  fill=False, edgecolor='red', linestyle='-', linewidth=1))
    con = ConnectionPatch(xyA=(T_max - 0.7, 0), coordsA=axis[0, 0].transData, xyB=(0, 0.75), coordsB=axis[0,1].transAxes,
                          arrowstyle="->", color="red", linewidth=2,connectionstyle="arc3,rad=-0.3")
    fig.add_artist(con)
    for i in range(C):
        eigen_state = Psi_eigenstate[(N * i):(N * (i + 1))]
        axis[0,2].scatter(omega[0, (N * i):(N * (i + 1))], eigen_state, color=color_list[i],s=15)
    axis[0, 2].set_xlim(-4.0, 4.0)
    axis[0, 2].set_ylim(-0.15, 0.15)
    axis[0,2].set_xlabel(r'${\omega}_{nodes}$', fontsize=12)
    axis[0,2].set_ylabel(r'${\theta}^{(\bar{E})}$', fontsize=12)
    axis[0,2].grid(False)

    # ------------------------------------------------------------------
    # DESD driven by a negative isolated Dirac eigestate(Edge)
    # ------------------------------------------------------------------
    Psi_T, Psi_dot_T, Psi_eigenstate, omega, E, eigen_matrix, eigen_Value, H_ham = DESD_Dynamic_function(B, Psi0,
                                                                                                         sigma, m=m,
                                                                                                         T_max=T_max,
                                                                                                         step_t=dt,
                                                                                                         label='neg')
    for i in range(len(Inter_edges)-1):
        for j in range(N*C+Inter_edges[i],N*C+Inter_edges[i+1]):
            theta_dot =Psi_dot_T[j,:].ravel()
            theta_dot = np.abs(theta_dot)
            index = dict_Inter_edges[i]
            axis[1,0].plot(time, theta_dot, color=color_list[i],linewidth=1.0)
            axis[1,1].plot(time[-1000:], theta_dot[-1000:], color=color_list[i],linewidth=1.0)

    axis[1, 1].set_ylim(-0.00, 0.17)
    axis[1, 0].set_xlabel(r'$t$', fontsize=12)
    axis[1, 0].set_ylabel(r'$\dot{\phi}_{hyper}$', fontsize=12)
    axis[1, 1].set_xlabel(r'$t$', fontsize=12)
    axis[1, 1].set_ylabel(r'$\dot{\phi}_{hyper}$', fontsize=12)
    axis[1, 1].tick_params(axis='x', labelsize=10)
    axis[1, 1].tick_params(axis='y', labelsize=10)
    axis[1, 0].grid(False)
    axis[1, 0].add_patch(
        Rectangle((T_max - 1, -0.5), 1,
                  1.2,
                  fill=False, edgecolor='red', linestyle='-', linewidth=1))
    con = ConnectionPatch(xyA=(T_max - 0.7, 0), coordsA=axis[1, 0].transData, xyB=(0.0, 0.75),
                          coordsB=axis[1, 1].transAxes,
                          arrowstyle="->", color="red", linewidth=2,connectionstyle="arc3,rad=-0.3")
    fig.add_artist(con)
    for i in range(len(Inter_edges)-1):
        eigen_state = Psi_eigenstate[N*C+Inter_edges[i]:N*C+Inter_edges[i+1]]
        index = dict_Inter_edges[i]
        axis[1, 2].scatter(omega[0, Inter_edges[i]:Inter_edges[i + 1]], np.abs(eigen_state), color=color_list[i], s=15)
    axis[1,2].set_xlim(-4.0,4.0)
    axis[1,2].set_ylim(-0.00,0.17)
    axis[1,2].set_xlabel(r'${\omega}_{hyper}$', fontsize=12)
    axis[1,2].set_ylabel(r'${\phi}^{(\bar{E})}$', fontsize=12)
    axis[1,2].grid(False)
    fig.tight_layout()
    plt.show()

