from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import make_interp_spline
from scipy.sparse import csr_matrix,bmat,eye
import random
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

# ================================================================
# Identify isolated eigenvalues in the Dirac spectrum
# ================================================================
def Calculate_IsolateValue(eigValue,eigVector,m,C=4,text='model1'):
    """
    Identify the isolated eigenvalue (above the mass gap +m) and
    extract its eigenvector.

    Returns
    -------
        pos_value : float
            Isolated positive eigenvalue
        pos_vector : array
            Its eigenvector
        sorted_Vals : array
            All sorted eigenvalues
        sorted_Vecs : array
            All sorted eigenvectors
    """
    # Sort eigenvalues
    sorted_indices = np.argsort(eigValue)
    sorted_Vals = np.real(eigValue[sorted_indices])
    sorted_Vecs = np.real(eigVector[:, sorted_indices])
    if text == 'model1':
        # Choose the first eigenvalue above +m
        pos_idx = np.where(sorted_Vals > m + 0.01)[0]
        # neg_idx = np.where(sorted_Vals < -(m + 0.01))[0][0]

        # position
        pos_value = sorted_Vals[pos_idx[0]]
        pos_vector = sorted_Vecs[:, pos_idx[0]]

        # negative
        # neg_value = sorted_Vals[neg_idx]
        # neg_vector = sorted_Vecs[neg_idx]

        return pos_value, pos_vector, sorted_Vals, sorted_Vecs
    elif text=='model2':
        # Gap-based model: search for large spectral gaps
        n = len(sorted_Vals)
        index_poses = np.where(sorted_Vals > m + 0.01)[0][:C]
        index_neges = np.where(sorted_Vals < m + 0.01)[0][-1*C:]
        index_pos = index_poses[0]
        index_neg = index_neges[0]
        err_pos = 0
        err_neg = 0
        for index in index_poses:
            evalue = sorted_Vals[index]
            # left gap
            if index == 0:
                delta_left = 0
            else:
                delta_left = evalue - sorted_Vals[index - 1]
            # right gap
            if index == n - 1:
                delta_right = 0
            else:
                delta_right = sorted_Vals[index + 1] - evalue
            if delta_left > err_pos and delta_right > err_pos and evalue > m+0.0001 :
                index_pos = index
                err_pos = min(delta_left,delta_right)
            if delta_left > err_neg and delta_right > err_neg and evalue < -(m+0.0001):
                index_neg = index
                err_neg = min(delta_left, delta_right)
        index_pos = np.where(sorted_Vals > m + 0.01)[0][0] if index_pos==0 else index_pos
        isolated_eigenvalues = np.array([sorted_Vals[index_pos],sorted_Vals[index_neg]])
        isolated_eigenvector = np.array([sorted_Vecs[:,index_pos],sorted_Vecs[:,index_neg]])
        # print(sorted_Vals[index_pos])
        pos_value = isolated_eigenvalues[isolated_eigenvalues>(m+0.01)][0]
        pos_vector = isolated_eigenvector[isolated_eigenvalues > (m + 0.01)][0]

        return pos_value,pos_vector,sorted_Vals,sorted_Vecs
    else:
        raise Exception("Unknown selection model.")

# ================================================================
# Dirac-Equation Synchronization Dynamics (DESD)
# ================================================================
def DESD_Dynamic_function(B,Psi0,sigma,m=1,C=4,T_max =20,step_t=0.01):
    """
        Complete DESD simulation on a directed hypergraph.

        Hypergraph represented by incidence matrix B:
            B(i,e) > 0 for head nodes
            B(i,e) < 0 for tail nodes

        Construct Dirac operator:
            D = [[0, B],
                 [Bᵀ, 0]]

            H = D + m Γ
            where Γ = diag(I_N , -I_M)

        Simulate:
            Ψ̇ = ω − σ Hᵀ sin(H Ψ)
        """
    N, M = B.shape
    total = N + M
    num_steps = int(T_max / step_t)
    Psi_T = np.zeros((total, num_steps + 1))
    Psi_dot_T = np.zeros((total, num_steps))
    D_Psi_T = np.zeros((total, num_steps + 1))

    # Build Dirac operator
    D = bmat([[csr_matrix((N, N)), B],
              [B.T, csr_matrix((M, M))]], format='csr')
    gamma = bmat([[eye(N, format='csr'), csr_matrix((N, M))],
                  [csr_matrix((M, N)), -eye(M, format='csr')]], format='csr')
    H = D + m * gamma

    # Full eigen decomposition
    eig_value, eig_vector = np.linalg.eigh(H.toarray())

    # Extract isolated eigenvalue & eigenvector
    E,Vector,eigValue,eigVector = Calculate_IsolateValue(eig_value,eig_vector,m,C-1,text='model2')
    pos = np.where(eigValue==E)

    Psi = Psi0.copy()
    plot_simple_dirac_spectrum(eigValue,m)
    # Natural frequencies for nodes & hyperedges
    Omega0, tau0 = 0, 1
    Omega1, tau1 = 0, 1
    w = Omega0 + np.random.randn(N,1) * tau0
    wedge = Omega1 + np.random.randn(M,1) * tau1

    Inv_eigvector = np.linalg.inv(eigVector)

    # Compute frequency in eigenbasis
    Omega = np.matmul(np.vstack([w, wedge]).T, Inv_eigvector).T
    pOmega = np.matmul(Inv_eigvector, Omega)

    # Amplify contribution of isolated eigenvalue
    pOmega[pos[0]] = 1

    omega = csr_matrix(eigVector @ pOmega)
    # Shift operator for dynamics
    Ham_E = H - E*eye(total)

    for i in range(num_steps):
        Psi_dot = Runge_Kutta(Ham_E,sigma,omega,Psi,step_t)

        Psi += Psi_dot*step_t

        Psi = csr_matrix(Psi)
        Psi_dot_T[:,i] = Psi_dot.toarray().squeeze()
        Psi_T[:,i+1] = Psi.toarray().squeeze()
        D_Psi_T[:,i] = (Psi_dot*step_t).toarray().squeeze()

    return Psi_T,Psi_dot_T,Vector,omega.toarray().T,E,eigVector,eigValue,Ham_E

# ================================================================
# Plot Dirac spectrum with density curve
# ================================================================
def plot_simple_dirac_spectrum(sorted_eig_vals, m=1):
    """
    Plot the density spectrum of eigenvalues and mark the mass gap ±m.
    """
    neglam = sorted_eig_vals[sorted_eig_vals < 0]
    nneglam = sorted_eig_vals[sorted_eig_vals >= 0]
    dnneglam = np.flip(nneglam)
    rho_neg = np.arange(1, len(neglam) + 1) / len(sorted_eig_vals)
    rho_nneg = np.arange(1, len(nneglam) + 1) / len(sorted_eig_vals)

    fig, ax_main = plt.subplots(figsize=(3.0, 2.49))
    for dx, dy, alpha in [(0.6, -0.3, 0.6), (0.8, -0.4, 0.3)]:
        ax_main.scatter(dnneglam + dx * 0.01, rho_nneg + dy * 0.01,
                    s=60, c="grey", alpha=alpha,
                    linewidths=0, zorder=2)
    ax_main.scatter(dnneglam, rho_nneg, c='#56B4E9', s=40, linewidths=0, zorder=3)
    ax_main.scatter(neglam, rho_neg, c='#56B4E9', s=40, linewidths=0, zorder=3)
    for dx, dy, alpha in [(0.6, -0.3, 0.6), (0.8, -0.4, 0.3)]:
        ax_main.scatter(neglam - dx * 0.01, rho_neg + dy * 0.01,
                        s=40, c="grey", alpha=alpha,
                        linewidths=0, zorder=2)

    ax_main.axvline(x=m, color='#3A5EA7', linestyle='--', linewidth=1.0)
    ax_main.axvline(x=-m, color='#3A5EA7', linestyle='--', linewidth=1.0)

    ax_main.set_xlabel('$E$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax_main.set_ylabel(r'$\rho_c(E)$', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.yticks(np.linspace(0, 1, 6))
    ax_main.tick_params(axis='x',labelsize=10)
    ax_main.tick_params(axis='y', labelsize=10)
    plt.grid(False)
    plt.show()

# ================================================================
# Visualization of directed hypergraph factor graph
# ================================================================
def visualize_factor_graph(hyper_edges, N, C,colors=None,C1=0,node_list = None):
    """
        Draw nodes arranged on a circle + diamond-shaped hyperedges.
    """
    node_colors = []
    if colors != None:
        C = C1
        colors = ['#A6CEE3', '#B2DF8A']
        for key, value in node_list.items():
            node_colors.append(colors[value])
        print(node_colors)
    else:
        colors = ['#777777']
        node_colors = [colors[n // N] for n in range(C * N)]
    G = nx.DiGraph()
    for i, he in enumerate(hyper_edges):
        edge_id = f"e{i}"
        G.add_node(edge_id, type="hyperedge")
        for t in he['tail']:
            G.add_edge(t, edge_id)
        for h in he['head']:
            G.add_edge(edge_id, h)

    pos = {}
    center_x = 0.0
    center_y = 0.0
    print(N * C)
    for j in range(N * C):
        node_id = j
        angle = 2 * np.pi * j / (N * C) - 2 * np.pi * 1.5 / (N * C)
        pos[node_id] = (center_x + 8 * np.cos(angle),
                        center_y + 8 * np.sin(angle))
    left_diamonds = [
        (-5.5, 1.0),
        (-5.5, -1.0),
        (-4.4, 2.6),
        (-4.6, -2.4),
        (-3.3, 3.4),
        (-3.1, 0.2),
        (-3.3, -3.1),
        (-2.6, 2.5),
        (-2.6, -2.3),
        (-1.8, -1.4),
        (-1.8, 1.3),
    ]
    right_diamonds = [
        (5.6, 1.2),
        (5.3, -1.1),
        (4.4, 2.5),
        (4.6, -2.6),
        (3.4, 3.2),
        (3.2, -0.2),
        (3.4, -3.1),
        (2.6, 2.5),
        (2.5, -2.3),
        (1.9, -1.2),
        (2.0, 1.3),
    ]
    for i, he in enumerate(hyper_edges):
        edge_id = f"e{i}"
        nodes = list(he['tail']) + list(he['head'])
        if len(nodes) > 0:
            # xs, ys = zip(*[pos[n] for n in nodes if np.random.rand(1)[0]<0.6])
            if i < 11:
                pos[edge_id] = right_diamonds[i]
            elif i < 22 and i >= 11:
                pos[edge_id] = left_diamonds[i - 11]
            else:
                pos[edge_id] = (0, 0)
    plt.close()
    plt.figure(1, figsize=(2.5, 1.8))
    hyper_nodes = [f"e{i}" for i in range(len(hyper_edges))]
    nx.draw_networkx_nodes(G, pos, nodelist=hyper_nodes,
                           node_color="#CAB2D6", edgecolors='#aaaaaa',
                           node_shape="D", node_size=30, alpha=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=range(C * N),
                           node_color=node_colors, edgecolors='#333333',
                           node_shape="o", node_size=150, alpha=0.6)
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color="gray",arrowstyle='->', alpha=0.7,connectionstyle=f'arc3,rad={0.15}')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    np.random.seed(33)

    B = np.array([[0.5, 0.0, 0.0, 0.0, -0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0.0, 0.5, -1.0, 0.5, -0.5, 0.5, -1.0, 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0.5, -1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 1.],
                  [-1.0, 0.5, 0.5, -1.0, 0.0, -1.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1., -1., 1.0, -1.0, 0.5, -1., 0.5, -0.5],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, -0.5, 0.0, 0., 0.5, 0.5, 0.],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, -1., 0.],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.5, -1., 0., 0., -0.5]])
    B = np.array([
        [-0.5, -0.5, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0],
        [1.0, 1.0, -1.0, -0.5, -1.0, -1.0, -1.0, 1.0, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0],
        [0.0, 0.0, 0.5, 1.0, 0.5, 0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0],
        [-0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.0, -0.5, 1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5, -0.5, 0.5, -1.0, -1.0, 0.0, 0.0, 0.5,
         0.5, -0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.5, 0.5, -1.0, 1.0, -1.0,
         0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0,
         -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
         0.0, -0.5]
    ])
    hyper_edges = []
    V,M = B.shape
    for i in range(M):
        edge = B[:,i]
        tail = []
        head = []
        for index,p in enumerate(edge):
            if p>0.0:
                head.append(index)
            elif p<0.0:
                tail.append(index)
        hyper_edge = {
            'tail':tail,
            'head':head
        }
        hyper_edges.append(hyper_edge)

    C = 2
    B = csr_matrix(B)
    num_edge = M
    N = int(V/C)
    # visualize_factor_graph(hyper_edges, N=V, C=1)

    Psi0 = 2 * np.pi*np.random.randn(V+num_edge,1)
    Psi0 = csr_matrix(Psi0)
    #initial param
    sigma = 10
    dt = 0.01
    T_max = 8
    m = 1
    colors = ['#E69F00', '#56B4E9', '#009E73', '#A6CEE3',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    Psi_T, Psi_dot_T, Psi_eigenstate, omega, E, eigen_matrix, eigen_Value, H_ham = DESD_Dynamic_function(B, Psi0, sigma,
                                                                                                         m=m,
                                                                                                         T_max=T_max,
                                                                                                           step_t=dt)
    plt.figure(2, figsize=(2.2, 1.8))
    label = ['A','B']
    for i in range(C):
        eigen_state = Psi_eigenstate[(N * i):(N * (i + 1))]
        plt.scatter(omega[0, (N * i):(N * (i + 1))], eigen_state, color=colors[i],s=80,edgecolors='k')
        if i == 0:
            plt.xlim(np.min(omega[0, :V]) - 0.2, np.max(omega[0, :V]) + 0.2)
            plt.ylim(np.min(Psi_eigenstate[:N*C]) - 0.1, np.max(Psi_eigenstate[:N*C]) + 0.1)
            plt.xlabel(r'${\omega}_{node}$', fontsize=10,labelpad=3)
            plt.ylabel(r'${\theta}^{(\bar{E})}$', fontsize=10,labelpad=3)

        plt.grid(False)
    plt.yticks(np.around(np.linspace(-0.75, 0.75, 5), 2), fontsize=10)
    plt.xticks(np.linspace(-1.5, 1.5, 5), fontsize=10)
    time = np.linspace(0, stop=T_max, num=int(T_max / dt) )
    fig, ax_main = plt.subplots(figsize=(2.2, 1.8))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch

    ax_inset = inset_axes(ax_main, width="50%", height="45%", bbox_to_anchor=(0.15, -0.25, 0.8, 0.8),
                          bbox_transform=ax_main.transAxes)
    ax_main.add_patch(
        Rectangle((T_max - 1, np.min(Psi_eigenstate[:N * C]) - 0.3), 1,
                  (np.max(Psi_eigenstate[:N * C]) - np.min(Psi_eigenstate[:N * C]) + 0.7),
                  fill=False, edgecolor='red', linestyle='--', linewidth=1))
    con = ConnectionPatch(
        xyA=(T_max - 0.2, 0), coordsA=ax_main.transData,
        xyB=(0.8, 1), coordsB=ax_inset.transAxes,
        arrowstyle="->", color="red", linewidth=2
    )
    fig.add_artist(con)
    for i in range(V):
        theta_dot = Psi_dot_T[i, :].ravel()
        eigen_state = Psi_eigenstate[(N * i):(N * (i + 1))]
        ax_main.plot(time, theta_dot, color=colors[i])
        ax_inset.plot(time[-100:],theta_dot[-100:], color=colors[i])
        ax_main.grid(False)
    # fig.tight_layout()
    ax_inset.set_yticks([-0.75,-0.38,0.0,0.38,0.75])
    ax_inset.tick_params(axis='x', labelsize=8)
    ax_inset.tick_params(axis='y', labelsize=8)
    ax_main.set_xlabel(r'$t$', fontsize=10, labelpad=3)
    ax_main.set_ylabel(r'$\dot{\theta}_{nodes}$', fontsize=10, labelpad=3)
    ax_main.set_yticks(np.around(np.linspace(-20, 20, 5), 1))
    ax_main.set_xticks([0,2,4,6,8])
    ax_main.tick_params(axis='x', labelsize=10)
    ax_main.tick_params(axis='y', labelsize=10)
    plt.show()
    # plt.figure(5)
    Psi_dot = Psi_dot_T[:,-1]
    mean = int(np.sum(Psi_dot[:N*C])/(N*C))
    node_list = {}
    C1 = 2
    for index, node_value in enumerate(Psi_dot[:N*C]):
        if node_value>mean:
            node_list[index]=0
        else:
            node_list[index] = 1
    visualize_factor_graph(hyper_edges, N=N, C=2,colors=colors,C1=2,node_list=node_list)
