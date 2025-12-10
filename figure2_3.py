from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix,bmat,eye
import scipy
from scipy.io import savemat, loadmat
"""
================================================================================
Dirac-Equation Synchronization Dynamics (DESD) on Directed Hypergraphs
--------------------------------------------------------------------------------
This script generates stochastic block model (SBM) hypergraphs, constructs the
Dirac operator H from the incidence matrix B, identifies isolated eigenstates,
and computes the nonlinear DESD evolution:
        Ψ̇ = ω − σ Hᵀ sin(H Ψ)
The code also visualizes the resulting Dirac eigenvalue spectrum and factor-graph
structure for community-level interpretation.
================================================================================
"""


# ============================================================
# 1. Isolated eigenvalue selection
# ============================================================
def Calculate_IsolateValue(eigValue,eigVector,m,C=4,text='model1',label='pos',id = 0):
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
        pos_value = sorted_Vals[pos_idx[id]]
        pos_vector = sorted_Vecs[:, pos_idx[id]]
        # negative
        neg_value = sorted_Vals[neg_idx[-1*(id+1)]]
        neg_vector = sorted_Vecs[:,neg_idx[-1*(id+1)]]
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
        # candidates on each side
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
def DESD_Dynamic_function(B,Psi0,sigma,m=1,C=4,T_max =20,step_t=0.01,label='pos',id = 0):
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
    E,Vector,eigValue,eigVector = Calculate_IsolateValue(eig_value,eig_vector,m,C-1,text='model1',label=label,id = id)
    pos = np.where(eigValue==E)
    Psi = Psi0.copy()

    # Generate random natural frequencies for nodes and hyperedges
    Omega0, tau0 = 0, 1
    Omega1, tau1 = 0, 1
    w = Omega0 + np.random.randn(N,1) * tau0
    wedge = Omega1 + np.random.randn(M,1) * tau1
    # Change basis to eigenbasis of H
    Inv_eigvector = np.linalg.inv(eigVector)

    Omega = np.matmul(np.vstack([w, wedge]).T, Inv_eigvector).T

    pOmega = np.matmul(Inv_eigvector, Omega)
    # Amplify contribution of isolated eigenmode
    pOmega[pos[0]] = 1
    # Back to original basis: natural frequency vector ω
    omega = csr_matrix(eigVector @ pOmega)
    # Shifted Hamiltonian H − E I
    Ham_E = H - E*eye(total)
    return Psi_T, Psi_dot_T, Vector, omega.toarray().T, E, eigVector, eigValue, Ham_E

# ============================================================
# 3. Plot Dirac eigenvalue density spectrum
# ============================================================
def plot_simple_dirac_spectrum(sorted_eig_vals, m=1):
    """
    Plot the cumulative (empirical) density of eigenvalues ρ_c(E)
    and highlight the mass gap around ±m.
    """
    neglam = sorted_eig_vals[sorted_eig_vals < 0]
    nneglam = sorted_eig_vals[sorted_eig_vals >= 0]
    dnneglam = np.flip(nneglam)
    rho_neg = np.arange(1, len(neglam) + 1) / len(sorted_eig_vals)
    rho_nneg = np.arange(1, len(nneglam) + 1) / len(sorted_eig_vals)

    plt.subplot(122)
    for dx, dy, alpha in [(0.6, -0.3, 0.6), (0.8, -0.4, 0.3)]:
        plt.scatter(dnneglam + dx * 0.01, rho_nneg + dy * 0.01,
                    s=60, c="grey", alpha=alpha,
                    linewidths=0, zorder=2)
    plt.scatter(dnneglam, rho_nneg, facecolors='#56B4E9', edgecolors='k', s=60,linewidths=0.0, zorder=3)

    plt.scatter(neglam, rho_neg, facecolors='#56B4E9', edgecolors='k',s=60,linewidths=0.0, zorder=3)
    for dx, dy, alpha in [(0.6, -0.3, 0.6), (0.8, -0.4, 0.3)]:
        plt.scatter(neglam - dx * 0.01, rho_neg + dy * 0.01,
                    s=60, c="grey", alpha=alpha,
                    linewidths=0, zorder=2)

    plt.xticks(np.linspace(-5, 5, 11),fontsize=10)
    plt.yticks(np.linspace(0, 1, 6), fontsize=10)
    # Vertical lines at ±m
    plt.axvline(x=m, color='#3A5EA7', linestyle='--', linewidth=1.0)
    plt.axvline(x=-m, color='#3A5EA7', linestyle='--', linewidth=1.0)
    plt.xlabel('$E$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.ylabel(r'$\rho_c(E)$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.grid(False)


def hypergraph_SBM(V, C, K, c1=15., c2=1.5,seed=33):
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
            tail_idx = rng.choice(
                range(len(nodes_h)),
                np.random.randint(1, len(nodes_h)),
                replace=False
            )

            hyper_edge = {
                'tail': list(nodes_h[tail_idx]),
                'head': list(np.delete(nodes_h, tail_idx))
            }
            hyper_edges.append(hyper_edge)
    # --------------------------------------------------------
    # Inter-community hyperedges
    # --------------------------------------------------------
    k = K[0]  # use first K as typical hyperedge size across communities
    Inter_num = len(hyper_edges)
    Inter_edges = [Inter_num]  # starting index for inter-community hyperedges
    dict_Inter_edges = {}
    num = 0  # index for inter-blocks
    nums = 0  # total number of inter-community hyperedges
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
            H_inter = int(round(c2*V*2/k))
            Inter_edges.append(Inter_edges[-1]+int(H_inter))
            dict_Inter_edges[num] = [i,j]
            num+=1
            nums += H_inter
            # half hyperedges oriented i → j
            for ij in range(int(H_inter/2)):
                k1 = np.random.randint(1,k)
                # k1 = max(1, k)
                k2 = k - k1
                idx_i = rng_i.choice(V, min(k1, V), replace=False)
                idx_j = rng_j.choice(V, min(k2, V), replace=False)
                nodes_h = np.concatenate([nodes_i[idx_i], nodes_j[idx_j]])
                hyper_edge = {
                    'tail': list(nodes_i[idx_i]),
                    'head': list(nodes_j[idx_j])
                }
                hyper_edges.append(hyper_edge)
            dict_Inter_edges[num] = [j,i]
            num+=1
            for ij in range(H_inter-int(H_inter/2)):
                k1 = np.random.randint(1,k)
                k2 = k - k1
                idx_i = rng_i.choice(V, min(k1, V), replace=False)
                idx_j = rng_j.choice(V, min(k2, V), replace=False)
                nodes_h = np.concatenate([nodes_i[idx_i], nodes_j[idx_j]])
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
    return hyper_edges, B, flag,Inter_edges,dict_Inter_edges,nums

# ============================================================
# 5. Factor-graph visualization
# ============================================================
def visualize_factor_graph(hyper_edges, N,intra_edges,node_list, C,num,colors=None,C1=0):
    """
       Visualize a directed hypergraph as a factor graph:
       - Circular communities of nodes
       - Diamond-shaped hyperedge nodes
       - Arrows from tail → hyperedge → head
    """
    node_colors=[]
    if colors != None:
        C = C1
        colors = ['#E69F00', '#009E73', '#F0E442', '#D55E00']
        for key,value in node_list.items():
                node_colors.append(colors[value])
    else:
        colors = ['#777777']
        node_colors = [colors[n // N] for n in range(C * N)]
    edges_colors=[]
    colors = ['#A6CEE3', '#56B4E9', '#B2DF8A',
                  '#F79A81', '#0072B2', '#B69CC0', '#CC79A7']
    for key,value in intra_edges.items():
            edges_colors.append(colors[value])
    G = nx.DiGraph()
    i=0
    # Add hyperedge nodes and their connections
    for key,value in intra_edges.items():
        he = hyper_edges[key]
        edge_id = f"e{i}"
        i+=1
        G.add_node(edge_id, type="hyperedge")
        for t in he['tail']:
            G.add_edge(t, edge_id)
        for h in he['head']:
            G.add_edge(edge_id, h)

    pos = {}
    radius = 22
    edge_poses = []
    # Place communities on a circle of radius `radius`
    for i in range(C):
        center_x = radius*np.cos(np.pi/4+i*2*np.pi/C)
        center_y = radius*np.sin(np.pi/4+i*2*np.pi/C)
        # Precompute midpoints between community centers (for hyperedges)
        for k in range(i+1,C):
            cen_x = radius * np.cos(np.pi / 4 + k * 2 * np.pi / C)
            cen_y = radius * np.sin(np.pi / 4 + k * 2 * np.pi / C)
            edge_poses.append([(center_x+cen_x)/2,(center_y+cen_y)/2])
        # Randomly distribute nodes around each community center
        for j in range(N):
            node_id = j+i*N
            pos[node_id] = (center_x + 7*np.random.random(), center_y + 7*np.random.random())
    i=0
    # Slight adjustments for edge positions to avoid overlap
    edge_poses[1] = [(a+b)/2 for a,b in zip(edge_poses[1],edge_poses[3])]
    edge_poses[4] = [(a+b)/2 for a,b in zip(edge_poses[4],edge_poses[2])]

    # Position hyperedge nodes near the midpoints between communities
    for key,value in intra_edges.items():
        he = hyper_edges[key]
        edge_id = f"e{i}"
        i+=1
        edge_pos = edge_poses[value]
        tail_nodes = list(he['tail'])
        head_nodes = list(he['head'])
        if len(tail_nodes)+len(head_nodes) > 0:
            xs = edge_pos[0]
            ys = edge_pos[1]
            pos[edge_id] = (xs+ 5*np.random.random(), + 5*np.random.random()+ys)

    plt.subplot(121)
    hyper_nodes = [f"e{i}" for i in range(len(intra_edges))]
    nx.draw_networkx_nodes(G, pos, nodelist=hyper_nodes,
                           node_color=edges_colors,edgecolors='black',linewidths=0.00,
                           node_shape="D", node_size=30, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=range(C*N),
                           node_color=node_colors,edgecolors='black',linewidths=0.00,
                           node_shape="o", node_size=50, alpha=1.0)
    shrink_value = 0
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color="gray",arrowstyle='->',width=0.5, alpha=0.8,connectionstyle=f'arc3,rad={0.0}',min_source_margin =shrink_value,min_target_margin=shrink_value)
    plt.axis("off")

# ============================================================
# 6. Main script: generate hypergraph, compute DESD, plot figures
# ============================================================
if __name__=='__main__':

    seed=23
    np.random.seed(seed)
    N = 20      # nodes per community
    C = 4       # number of communities
    dt = 0.01   # time step
    T_max = 10  # horizon
    m = 1.0     # mass term in Dirac operator
    sigma = 10  # coupling strength
    sigmas = np.arange(0, 10, step=1)  # optional sweep over couplings
    # hyperedge size per community (here all =3, but length=10 unused)
    k = list(np.ones(10,dtype=int)*3)

    # --------------------------------------------------------
    # Generate hypergraph SBM instance
    # --------------------------------------------------------
    hyper_edges, B, flag,Inter_edges,dict_Inter_edges,nums = hypergraph_SBM(N,C,k,c1=50,c2=1.,seed = seed)
    nrun = 0
    while flag==0 and nrun<20:
        hyper_edges, B, flag,Inter_edges,dict_Inter_edges,nums = hypergraph_SBM(N,C,k,c1=50,c2=1.,seed=seed)
        nrun+=1

    num_edge = len(hyper_edges)
    # Random initialization of node + hyperedge phases Ψ(0)
    Psi0 = 2 * np.pi*np.random.randn(N*C+num_edge,1)
    Psi0 = csr_matrix(Psi0)

    time = np.linspace(0, stop=T_max, num=int(T_max / dt))
    # --------------------------------------------------------
    # Figure 2: factor graph + Dirac spectrum
    # --------------------------------------------------------
    fig, axis = plt.subplots(1, 2, figsize=(5.4, 2.56))
    sub_fig_labels = ['(a)', '(b)']
    fig.tight_layout()
    for id in range(len(sub_fig_labels)):
        pos = axis[id].get_position()
        fig.text(
            pos.x0-0.05*id-0.03,
            pos.y1+0.01*id,
            sub_fig_labels[id],
            fontsize=12,
            va='bottom',
            ha='right'
        )
    # Compute DESD spectral objects (positive isolated eigenvalue)
    Psi_T, Psi_dot_T, Psi_eigenstate, omega, E,eigen_matrix,eigen_Value,H_ham = DESD_Dynamic_function(B, Psi0, sigma, m=m, T_max=T_max, step_t=dt,label='pos',id=0)
    # Plot Dirac spectrum on right panel
    plot_simple_dirac_spectrum(eigen_Value, m)
    # Node → community mapping
    node_list = {i:j for j in range(C) for i in range(j*N,(j+1)*N)}
    Community_intra= {i: j for j in range(len(Inter_edges)-1) for i in range(Inter_edges[j], Inter_edges[j + 1])}

    color_list = ['#A6CEE3', '#56B4E9', '#B2DF8A',
                  '#F79A81', '#0072B2', '#B69CC0', '#CC79A7']
    # Factor-graph visualization on left panel
    visualize_factor_graph(hyper_edges,num=nums,N=N, C=C,colors=color_list,C1=C,node_list=node_list,intra_edges = Community_intra)
    fig.tight_layout()
    plt.show()

    fig, axis = plt.subplots(2, 3, figsize=(7, 3.8))
    fig.tight_layout()
    sub_fig_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
    for label, ax in zip(sub_fig_labels, axis.flat):
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.01,
            pos.y1 + 0.01,
            label,
            fontsize=12,
            va='bottom',
            ha='right'
        )
    #cluster classification with different positive and negative isolated eigenstate
    for id in range(3):
        np.random.seed(seed)
        color_list = ['#A6CEE3', '#56B4E9', '#B2DF8A',
                      '#F79A81', '#0072B2', '#B69CC0', '#CC79A7']
        colors = ['#E69F00', '#009E73', '#F0E442', '#D55E00']
        time = np.linspace(0, stop=T_max, num=int(T_max / dt))
        # Positive isolated eigenvalue: projection on nodes
        Psi_T, Psi_dot_T, Psi_eigenstate, omega, E, eigen_matrix, eigen_Value, H_ham = DESD_Dynamic_function(B, Psi0,
                                                                                                             sigma, m=m,
                                                                                                             T_max=T_max,
                                                                                                             step_t=dt,
                                                                                                             label='pos',
                                                                                                             id=id)

        for i in range(C):
            eigen_state = Psi_eigenstate[(N * i):(N * (i + 1))]
            axis[0, id].scatter(omega[0, (N * i):(N * (i + 1))], eigen_state, color=colors[i], s=15)
        axis[0, id].set_xlim(-3.5,3.5)
        axis[0, id].set_ylim(-0.2,0.25)
        axis[0, id].set_xlabel(r'${\omega}_{nodes}$', fontsize=12)
        axis[0, id].set_ylabel(r'${\theta}^{(\bar{E})}$', fontsize=12)
        axis[0, id].grid(False)

        # Negative isolated eigenvalue: projection on nodes
        Psi_T, Psi_dot_T, Psi_eigenstate, omega, E, eigen_matrix, eigen_Value, H_ham = DESD_Dynamic_function(B, Psi0,
                                                                                                             sigma, m=m,
                                                                                                             T_max=T_max,
                                                                                                             step_t=dt,
                                                                                                             label='neg',
                                                                                                             id=id)
        i=0
        while i<len(Inter_edges)-1:
            eigen_state = Psi_eigenstate[N * C + Inter_edges[i]:N * C + Inter_edges[i + 1]]
            axis[1, id].scatter(omega[0, Inter_edges[i]:Inter_edges[i + 1]], np.abs(eigen_state), color=color_list[int(i)], s=15)
            i += 1
        axis[1, id].set_xlim(-3.5, 3.5)
        axis[1, id].set_ylim(-0.02,0.2)
        axis[1, id].set_xlabel(r'${\omega}_{hyper}$', fontsize=12)
        axis[1, id].set_ylabel(r'${\phi}^{(\bar{E})}$', fontsize=12)
        axis[1, id].grid(False)
    fig.tight_layout()

    plt.show()