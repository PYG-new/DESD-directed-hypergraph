import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# Build a directed hypergraph from the high-school contact data
# (PC, PC*, MP*1, MP*2), compute the incidence matrix B,
# community dictionary, and inter-community hyperedges.
# Also plot the node total-degree distribution and save to .npz.
# ============================================================
def plot_hypergraph_total_degree(directed_hyperedges,node_T_index, N, title_suffix=""):
    """
    Plot the total degree (in-degree + out-degree) distribution
    for all nodes in a directed hypergraph.
    """
    # Initialize total degree array
    total_degrees = np.zeros(N)
    # Accumulate degree contribution from each directed hyperedge
    for hyper_edge in directed_hyperedges:
        tail = hyper_edge['tail']
        head = hyper_edge['head']
        tail_len = len(tail)
        head_len = len(head)

        # Out-going contribution from tail nodes
        if tail_len > 0:
            weight = 1 / tail_len
            for t in tail:
                total_degrees[node_T_index[t]] += weight

        # In-coming contribution from head nodes
        if head_len > 0:
            weight = 1 / head_len
            for h in head:
                total_degrees[node_T_index[h]] += weight
    # Use IQR-based rule to choose histogram bin width
    q75, q25 = np.percentile(total_degrees, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        # If all degrees are similar, fall back to a simple rule
        bins = max(10, int(N / 10))
    else:
        # Freedman–Diaconis-like rule with an extra /2 to refine bins
        bin_width = (2 * iqr) / (len(total_degrees) ** (1 / 3)) / 2
        bins = int((total_degrees.max() - total_degrees.min()) / bin_width)
        bins = max(bins, 10)
    # Plot histogram of total degree
    plt.figure(figsize=(8, 5))
    plt.hist(total_degrees, bins=bins, color='#56B4E9', alpha=0.7, edgecolor='black', density=True)
    plt.title(f'Node degree distribution{title_suffix}', fontsize=12, fontweight='bold')
    plt.xlabel('Node degree (indegree + outdegree)')
    plt.ylabel('Probability density')
    # KDE smoothing curve (if variance is non-zero)
    if np.std(total_degrees) > 0:
        kde = gaussian_kde(total_degrees)
        x_range = np.linspace(total_degrees.min(), total_degrees.max(), 200)
        plt.plot(x_range, kde(x_range), color='red', linewidth=2.5,
                 label='Probability density fitting curve')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__=='__main__':
    dir = './high_school/'
    # --------------------------------------------------------
    # 1. Load raw hyperedge list from text
    # --------------------------------------------------------
    hypergraph_data = []
    with open(dir+"hyperedges.txt",'r',encoding="utf-8") as f:
        for index,line in enumerate(f,1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            fields = np.array(fields, dtype=int)
            hypergraph_data.append(fields)
    print(f'num of hyperedges {len(hypergraph_data)}')
    # --------------------------------------------------------
    # 2. Load community labels and node-to-community mapping
    # --------------------------------------------------------
    Community_label = []
    with open(dir+"label_names.txt",'r',encoding="utf-8") as f:
        for index,line in enumerate(f,1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            Community_label.append(fields)
    print(f'num of community {len(Community_label)}')
    node_labels = []
    with open(dir+"node_labels.txt",'r',encoding="utf-8") as f:
        for index,line in enumerate(f,1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            fields = np.array(fields, dtype=int)
            node_labels.append(fields)
    print(f'num of person {len(node_labels)}')
    # --------------------------------------------------------
    # 3. Select target communities (here PC / PC* / MP*1 / MP*2)
    # --------------------------------------------------------
    target = ['PC','PC*','MP*1','MP*2']
    target_idx = {label:[] for label in target}
    # Map each target community label to its community index (1-based)
    for index,label in enumerate(Community_label):
        if label[0] in target:
            target_idx[label[0]].append(index+1)
    indexes = [i[0] for i in list(target_idx.values())]

    # --------------------------------------------------------
    # 4. Restrict to students belonging to these communities
    # --------------------------------------------------------
    N = len(node_labels)
    # Keep only nodes whose community index is in `indexes`
    student_nodes = [n+1 for n in range(0, N) if node_labels[n].item() in indexes]
    # Map original node ID → compact index [0, N_sub-1]
    node_to_idx = {node: idx for idx, node in enumerate(student_nodes)}
    # Map label string → community index among the selected targets
    label_to_idx = {label[0]:idx for idx,label in enumerate(Community_label) if label[0] in target}
    # --------------------------------------------------------
    # 5. Convert undirected hyperedges to directed hyperedges
    #    (random tail/head split), and identify inter-community edges
    # --------------------------------------------------------
    nums_edge = len(hypergraph_data)
    hyper_edges = []
    # Dictionary of inter-community hyperedges by label-pair
    inter_hyperedge_Com = {f'{list(label_to_idx.keys())[i]}-{list(label_to_idx.keys())[j]}':[] for i in range(len(target)) for j in range(i+1,len(target))}
    for i in range(nums_edge):
        edge = np.array(hypergraph_data[i])
        # Community indices of nodes in this hyperedge
        labels = [node_labels[indx-1][0] for indx in edge]

        num = edge.shape[0]
        # Distinct community indices present in this hyperedge
        labels = sorted(list(set(labels)))
        # Only keep hyperedges fully contained in student_nodes
        common = np.intersect1d(edge, student_nodes)
        if common.size==num:
            # Randomly choose tail/head partition
            tail_idx = np.random.choice(range(num),np.random.randint(1,num),replace=False)
            hyper_edge = {
                'tail':list(edge[tail_idx]),
                'head':list(np.delete(edge,tail_idx))
            }
            hyper_edges.append(hyper_edge)
            # If this hyperedge connects exactly two communities, store its ID
            if len(labels)==2:
                print(labels)
                idx=len(hyper_edges)
                inter_hyperedge_Com[f'{Community_label[labels[0]-1][0]}-{Community_label[labels[1]-1][0]}'].append(idx-1)
        else:
            continue
    N = len(student_nodes)
    M = len(hyper_edges)

    # --------------------------------------------------------
    # 6. Build community dictionary over the filtered node set
    # --------------------------------------------------------
    Community = {list(target_idx.keys())[i]:[] for i in range(len(target_idx))}
    for i in student_nodes:
        idx = node_labels[i-1][0]
        label = Community_label[idx-1][0]
        Community[label].append(node_to_idx[i])

    # --------------------------------------------------------
    # 7. Build incidence matrix B (N × M) for directed hypergraph
    # --------------------------------------------------------
    B = np.zeros((N, M))
    for l, hyper_edge in enumerate(hyper_edges):
        tail_len = len(hyper_edge['tail'])
        head_len = len(hyper_edge['head'])
        for t in hyper_edge['tail']:
            B[node_to_idx[t], l] = -1 / tail_len
        for h in hyper_edge['head']:
            B[node_to_idx[h], l] = 1 / head_len
    node_degrees = np.sum(np.abs(B), axis=1)
    non_isolated_nodes = np.where(node_degrees == 0)[0]
    name = "subset_hypergraph_N{}_M{}_C{}_4".format(N, M, len(Community))
    plot_hypergraph_total_degree(hyper_edges,node_to_idx, N)
    if len(non_isolated_nodes)==0:
        np.savez(name + '.npz', B=B, hyper_edges=hyper_edges, Community=Community,node_to_idx=node_to_idx,inter_hyperedge=inter_hyperedge_Com)
        print('save successful')
