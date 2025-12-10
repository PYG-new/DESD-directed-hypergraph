import time
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix,bmat,eye
import networkx as nx
from scipy.io import savemat, loadmat
from scipy.sparse.linalg import eigsh
"""
============================================================
Dirac-equation synchronization dynamics on a directed hypergraph
------------------------------------------------------------
- Load an empirical directed hypergraph (high-school subset).
- Build Dirac operator H from the incidence matrix B.
- Identify an isolated Dirac eigenvalue E and its eigenstate.
- Construct a natural-frequency vector ω concentrated near E.
- Integrate Dirac synchronization dynamics via RK4:
      Ψ̇ = ω − σ Hᵀ sin(H Ψ)
- Visualize node/edge time derivatives and projections
  onto eigenstates around E.
============================================================
================================================================
1. Runge–Kutta 4 for integrating Dirac synchronization dynamics
================================================================
"""
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
# 3. Dirac-Equation Synchronization Dynamics (DESD)
# ============================================================
def DESD_Dynamic_function(B,sigma,m=1,C=4,T_max =20,step_t=0.01,label='pos'):
    """
    Compute Dirac-Equation Synchronization Dynamics (DESD) on a directed hypergraph.

    Steps:
        (1) Build Dirac operator H from incidence matrix B.
        (2) Identify isolated eigenmode ±E.
        (3) Construct natural frequencies ω in the Dirac basis.
        (4) Integrate Ψ̇ = ω − σ Hᵀ sin(H Ψ) using RK4.
    """
    N, M = B.shape  # #nodes, #hyperedges
    total = N + M   # dimension of topological signals (nodes + hyperedges)
    num_steps = int(T_max / step_t)
    Psi0 = np.random.randn(total,1)
    Psi0 = csr_matrix(Psi0)

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
    E,Vector,eigValue,eigVector = Calculate_IsolateValue(eig_value,eig_vector,m,C-1,text='model1',label=label)
    pos = np.where(eigValue==E)
    poses = np.where((eigValue < E) & (eigValue > m + 0.001))
    print(E)
    Psi = Psi0.copy()
    Psi_T[:,0] = Psi.toarray().squeeze()
    if label =='pos':
        plot_simple_dirac_spectrum(eigValue,m)
    # Generate random natural frequencies for nodes and hyperedges
    Omega0, tau0 = 0, 1
    Omega1, tau1 = 0, 1
    w = Omega0 + np.random.randn(N,1) * tau0
    wedge = Omega1 + np.random.randn(M,1) * tau1
    # Change basis to eigenbasis of H
    Inv_eigvector = np.linalg.inv(eigVector)

    Omega = np.matmul(np.vstack([w, wedge]).T, Inv_eigvector).T

    pOmega = np.matmul(Inv_eigvector, Omega)
    id_pos = C - len(poses)
    pOmega[pos[0]] = 1
    alpha = 1.0
    for k in range(1, len(poses) + 1):
        pOmega[pos[0] - k] = pOmega[pos[0] - k] * alpha
    for k in range(1, id_pos + 1):
        pOmega[pos[0] + k] = pOmega[pos[0] + k] * alpha
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
    ax_main.set_title('Eigenvalue density spectrum',fontsize=12, fontname="Times New Roman")
    ax_main.scatter(dnneglam, rho_nneg, facecolors='none', color='blue')
    ax_main.scatter(neglam, rho_neg, facecolors='none', color='blue')
    ax_main.add_patch(Rectangle((np.min(dnneglam)-0.1,-0.01), np.max(dnneglam)-np.min(dnneglam)+0.2,np.max(rho_nneg)+0.02,fill=False,edgecolor='red', linestyle='--', linewidth=2))
    ax_main.set_xlabel('$E$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax_main.set_ylabel(r'$\rho_c(E)$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax_inset = inset_axes(ax_main, width="50%", height="50%", loc='upper right')
    ax_inset.scatter(dnneglam, rho_nneg,facecolors='none', color='red')
    ax_inset.tick_params(axis='x', labelsize=10)
    ax_inset.tick_params(axis='y', labelsize=10)
    plt.grid(False)
    plt.show()
if __name__=='__main__':
    np.random.seed(44)
    np.random.seed(32)

    colors_20 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
        '#8c564b','#e377c2','#7f7f7f','#bcbd22',
        '#17becf','#393b79','#637939','#8c6d31',
        '#843c39','#7b4173','#3182bd','#31a354',
        '#756bb1','#636363','#e6550d',
    ]
    # --------------------------------------------------------
    # Load preprocessed high-school hypergraph (subset)
    # --------------------------------------------------------
    data = np.load("./subset_hypergraph_N150_M3160_C4_4.npz", allow_pickle=True)
    hyper_edges = list(data['hyper_edges'])
    B = csr_matrix(data['B'])
    node_to_idx = data['node_to_idx'].item()
    inter_hyperedge_Com = data['inter_hyperedge'].item()
    community =data['Community'].item()

    # ------------------------------------------------------------------
    # Hypergraph and dynamics parameters
    # ------------------------------------------------------------------
    T_max=5     # horizon
    dt=0.001    # time step
    sigma=4.0   # coupling strength (used in full dynamic version)
    m=1         # mass term in Dirac operator
    N,M = B.shape
    C = len(community)
    time = np.linspace(0, stop=T_max, num=int(T_max / dt))
    # ------------------------------------------------------------------
    # DESD driven by a positive isolated Dirac eigestate(Nodes)
    # ------------------------------------------------------------------
    Psi_T, Psi_dot_T, Psi_eigenstate, omega, E, eigen_matrix, eigen_Value, H_ham = DESD_Dynamic_function(B, sigma,
                                                                                                         m=m,
                                                                                                         T_max=T_max,
                                                                                                         step_t=dt,
                                                                                                         label='pos')
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch

    i, i_A, i_B = 0, 0, 0
    # ------------------------------------------------------------------
    # Figure with 2 × 3 subpanels:
    #   Row 1: node-level dynamics and eigenstate projection
    #   Row 2: hyperedge-level dynamics and eigenstate projection
    # ------------------------------------------------------------------
    fig, axis = plt.subplots(2, 3, figsize=(7, 3.8))
    fig.tight_layout()
    sub_fig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)','(g)']
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
    for key, value in community.items():
        eigen_state = Psi_eigenstate[value]
        axis[0,2].scatter(omega[0, value], eigen_state, color=colors_20[i],s=15)
        i += 1
    axis[0,2].set_xlim(-3.0,3.0)
    axis[0,2].set_ylim(-0.15,0.15)
    axis[0,2].set_xlabel(r'${\omega}_{nodes}$', fontsize=12)
    axis[0,2].set_ylabel(r'${\theta}^{(\bar{E})}$', fontsize=12)
    axis[0,2].grid(False)
    axis[0,0].add_patch(
        Rectangle((T_max - 1, -3.0), 1,
                  6.0,
                  fill=False, edgecolor='red', linestyle='-', linewidth=1))
    con = ConnectionPatch(xyA=(T_max - 0.7, 0), coordsA=axis[0,0].transData, xyB=(0.0, 0.75),
                          coordsB=axis[0,1].transAxes,
                          arrowstyle="->", color="red", linewidth=2, connectionstyle="arc3,rad=-0.3")
    fig.add_artist(con)
    i, i_A, i_B = 0, 0, 0
    for key, value in community.items():
        for j in value:
            theta_dot = Psi_dot_T[j, :].ravel()
            axis[0,0].plot(time, theta_dot, color=colors_20[i],linewidth=1.0)
            axis[0,1].plot(time[-1000:], theta_dot[-1000:], color=colors_20[i],linewidth=1.0)
        i+=1
    axis[0, 1].set_ylim(-0.15, 0.15)
    axis[0, 0].set_ylim(-50, 50)
    axis[0,0].set_xlabel(r'$t$', fontsize=12)
    axis[0,0].set_ylabel(r'$\dot{\theta}_{nodes}$', fontsize=12)
    axis[0,1].set_xlabel(r'$t$', fontsize=12)
    axis[0,1].set_ylabel(r'$\dot{\theta}_{nodes}$', fontsize=12)
    axis[0,0].grid(False)
    axis[0,1].grid(False)
    # ------------------------------------------------------------------
    # DESD driven by a negative isolated Dirac eigestate(Edge)
    # ------------------------------------------------------------------
    Psi_T, Psi_dot_T, Psi_eigenstate, omega, E, eigen_matrix, eigen_Value, H_ham = DESD_Dynamic_function(B, sigma,
                                                                                                         m=m,
                                                                                                         T_max=T_max,
                                                                                                         step_t=dt,
                                                                                                         label='neg')
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch

    i, i_A, i_B = 0, 0, 0
    for key, value in inter_hyperedge_Com.items():
        for j in value:
            eigen_state = Psi_eigenstate[N+j]
            eigen_state = np.abs(eigen_state)
            axis[1,2].scatter(omega[0, j], eigen_state, color=colors_20[i],s=15)
        i+=1
    axis[1,2].set_xlim(-3.0,3.0)
    axis[1,2].set_ylim(-0.03,0.15)
    axis[1,2].set_xlabel(r'${\omega}_{hyper}$', fontsize=12)
    axis[1,2].set_ylabel(r'${\phi}^{(\bar{E})}$', fontsize=12)
    axis[1,2].grid(False)
    axis[1,0].add_patch(
        Rectangle((T_max - 1, -0.25), 1,
                  0.65,
                  fill=False, edgecolor='red', linestyle='-', linewidth=1))
    con = ConnectionPatch(xyA=(T_max - 0.7, 0), coordsA=axis[1,0].transData, xyB=(0.0, 0.75),
                          coordsB=axis[1,1].transAxes,
                          arrowstyle="->", color="red", linewidth=2, connectionstyle="arc3,rad=-0.3")
    fig.add_artist(con)
    i, i_A, i_B = 0, 0, 0
    for key, value in inter_hyperedge_Com.items():
        for j in value:
            theta_dot = Psi_dot_T[N+j, :].ravel()
            theta_dot = np.abs(theta_dot)
            axis[1,0].plot(time, theta_dot, color=colors_20[i],linewidth=1.0)
            axis[1,1].plot(time[-1000:], theta_dot[-1000:], color=colors_20[i],linewidth=1.0)

        i+=1
    axis[1,0].set_xlabel(r'$t$', fontsize=12)
    axis[1,1].set_ylim(-0.03,0.15)
    axis[1,0].set_ylabel(r'$\dot{\phi}_{hyper}$', fontsize=12)
    axis[1,1].set_xlabel(r'$t$', fontsize=12)
    axis[1,1].set_ylabel(r'$\dot{\phi}_{hyper}$', fontsize=12)
    axis[1,0].grid(False)
    axis[1,1].grid(False)
    fig.tight_layout()
    plt.show()



