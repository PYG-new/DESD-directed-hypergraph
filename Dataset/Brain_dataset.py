"""
Construct directed hyperedges from Yeo-7 network–based functional connectivity
(ABIDE fMRI), and save a directed hypergraph in incidence-matrix form.

Pipeline:
    1. Load Schaefer-200 atlas with Yeo-7 networks and select target modules.
    2. Load ABIDE subjects and extract ROI time series using NiftiLabelsMasker.
    3. Compute subject-level functional connectivity (Pearson correlation).
    4. Within each module: threshold by strength percentiles to create
       intra-module directed hyperedges (tail/head split is random).
    5. Between modules: threshold inter-module connectivity to create
       inter-module directed hyperedges.
    6. Optionally visualize the directed factor graph (nodes + hyperedges).
    7. Build incidence matrix B (nodes × hyperedges) and save all data to .npz.
"""

from collections import defaultdict
# from nilearn import input_data, datasets
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn import datasets
import itertools
from nilearn.maskers import NiftiLabelsMasker
import matplotlib.pyplot as plt
def download_yeo_7_network_data():
    """
        Build directed hyperedges from Schaefer-200/Yeo-7 atlas and ABIDE fMRI data.
    """
    # ------------------------------------------------------------------
    # 1) Load Schaefer-200 atlas with Yeo-7 network labels
    # ------------------------------------------------------------------
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    labels = atlas.labels  # list of ROI label strings
    atlas_filename = atlas.maps # 4D NIfTI atlas
    # Parse ROI labels into functional modules (communities)
    communities = defaultdict(list)
    id = 0      # compressed node index (for selected ROIs)
    id_ = 0     # original ROI index (0..199)
    filter_name = ['Default','Vis','SomMot']
    map_T_id = defaultdict(list)    # original index -> compressed index
    for lab in labels:
        if '7Networks' in lab:
            part = lab.split('_')
            name = ""
            name+=part[2]

            # Skip ROIs not in the filtered networks
            if name not in filter_name:
                id_ += 1
                map_T_id[id_] = -1
                continue
            # Add original ROI index to this module
            communities[name].append(id_)
            # Map original index -> compressed index
            map_T_id[id_]=id
            id+=1
            id_+=1
        else:
            print(f"{lab} not in 7networks")
    num_nodes = id
    print(f"num nodes: {num_nodes}, module nums: {len(communities)}")

    # ------------------------------------------------------------------
    # 2) Load ABIDE fMRI data (first num_subjects subjects)
    # ------------------------------------------------------------------
    num_subjects = 7
    abide = datasets.fetch_abide_pcp(
        n_subjects=num_subjects,
        pipeline='cpac',
        derivatives=['func_preproc'],
        data_dir="../nilearn_data/ABIDE_pcp"
    )
    hyperedges = []         # list of all directed hyperedges
    id = 0                  # global hyperedge ID counter
    inter_hyperedge = []    # per inter-module pair: list of hyperedge IDs
    # ------------------------------------------------------------------
    # 3) Loop over subjects: build intra- & inter-module hyperedges
    # ------------------------------------------------------------------
    for k in range(num_subjects):
        print(f'eporch:{k}')
        inter_id =0     # index over inter-module pairs for this subject
        fmri_file = abide.func_preproc[k]
        # Extract ROI time series using the atlas
        masker = NiftiLabelsMasker(
            labels_img=atlas_filename,
            standardize=True,
            detrend=True,
            t_r=2.0,
            low_pass=0.1,
            high_pass=0.01
        )
        time_series = masker.fit_transform(fmri_file)

        # Functional connectivity (Pearson correlation)
        corr_measure = ConnectivityMeasure(kind='correlation')
        fc_matrix = corr_measure.fit_transform([time_series])[0]
        # --------------------------------------------------------------
        # 3.1 Intra-module hyperedges (within each functional module)
        # --------------------------------------------------------------
        for name, nodes in communities.items():
            sub_fc = fc_matrix[np.ix_(nodes, nodes)]
            n = len(nodes)
            # Use positive upper-triangular entries as connection strengths
            strengths = sub_fc[np.triu_indices_from(sub_fc, k=1)]
            strengths = strengths[strengths > 0]

            if len(strengths) == 0:
                continue

            # Discretize connection strengths into percentile bins
            percentile_points = np.linspace(40, 100, 7)
            bin_edges = np.percentile(strengths, percentile_points)
            strength_bins = list(zip(bin_edges, bin_edges[1:]))

            # For each node, create hyperedges based on bins
            for i in range(n):
                for lower_bound, upper_bound in strength_bins:
                    if upper_bound == bin_edges[-1]:
                        condition = (sub_fc[i, :] >= lower_bound) & (sub_fc[i, :] <= upper_bound)
                    else:
                        condition = (sub_fc[i, :] >= lower_bound) & (sub_fc[i, :] < upper_bound)

                    connected_indices = np.where(condition)[0]
                    # Create a directed hyperedge if at least 2 nodes are connected
                    if len(connected_indices) >= 2:
                        edge_nodes = np.array(list(set([nodes[i]] + [nodes[j] for j in connected_indices])))
                        tail_idx = np.random.choice(range(len(edge_nodes)),np.random.randint(1,len(edge_nodes)),replace=False)
                        hyperedges.append({
                            'module': name,     # intra-module edge
                            'id':id,
                            'tail': list(edge_nodes[tail_idx]),
                            'head': list(np.delete(edge_nodes,tail_idx))
                        })
                        id+=1
        # --------------------------------------------------------------
        # 3.2 Inter-module hyperedges (between pairs of modules)
        # --------------------------------------------------------------
        inter_percentile_points = np.linspace(80, 100, 2)
        # inter_hyperedge.append(id)
        label = []
        for m1,idx1 in communities.items():
            label.append(m1)
            for m2, idx2 in communities.items():
                # avoid double counting (m1-m2 and m2-m1)
                if m2 in label:
                    continue
                idx1, idx2 = communities[m1], communities[m2]

                # (m1, m2) block: inter-module connectivity
                sub_fc_inter = fc_matrix[np.ix_(idx1, idx2)]
                strengths_ter = sub_fc_inter[np.triu_indices_from(sub_fc_inter, k=1)]
                strengths_ter = strengths_ter[strengths_ter > 0]
                bin_edges_ter = np.percentile(strengths_ter, inter_percentile_points)
                strength_bins_ter = list(zip(bin_edges_ter, bin_edges_ter[1:]))

                # (m1, m1) block: intra-module connectivity for m1
                sub_fc_intra = fc_matrix[np.ix_(idx1, idx1)]
                strengths_tra = sub_fc_intra[np.triu_indices_from(sub_fc_intra, k=1)]
                strengths_tra = strengths_tra[strengths_tra > 0]
                bin_edges_tra = np.percentile(strengths_tra, inter_percentile_points)
                strength_bins_tra = list(zip(bin_edges_tra, bin_edges_tra[1:]))

                inter_hyper = []
                # For each node in m1, build a cross-module hyperedge
                for i in range(len(idx1)):
                    connections_to_tra = sub_fc_intra[i, :]
                    connections_to_ter = sub_fc_inter[i, :]
                    for tra_bin,ter_bin in zip(strength_bins_tra,strength_bins_ter):
                        lower_tra, upper_tra = tra_bin
                        lower_ter, upper_ter = ter_bin

                        # Threshold intra-module connections
                        if upper_tra == bin_edges_ter[-1]:
                            condition_tra = (connections_to_tra >= lower_tra) & (connections_to_tra <= upper_tra)
                        else:
                            condition_tra = (connections_to_tra >= lower_tra) & (connections_to_tra < upper_tra)
                        # Threshold inter-module connections
                        if upper_ter == bin_edges_tra[-1]:
                            condition_ter = (connections_to_ter >= lower_ter) & (connections_to_ter <= upper_ter)
                        else:
                            condition_ter = (connections_to_ter >= lower_ter) & (connections_to_ter < upper_ter)

                        connected_tra = np.where(condition_tra)[0]
                        connected_ter = np.where(condition_ter)[0]

                        nodes_tra = np.array(idx1)[connected_tra]
                        nodes_ter = np.array(idx2)[connected_ter]

                        has_inter_id1 = nodes_tra.size > 0
                        has_inter_id2 = nodes_ter.size > 0
                        # Only keep hyperedges that connect both modules
                        if has_inter_id2 and has_inter_id1:
                            hyperedges.append({
                                "modules": m1+'-'+m2,
                                "id":id,
                                "tail": list(nodes_tra),
                                "head": list(nodes_ter),
                            })
                            inter_hyper.append(id)
                            id+=1
                # Accumulate inter-module hyperedge IDs across subjects
                if k==0:
                    inter_hyperedge.append(inter_hyper)
                else:
                    inter_hyperedge[inter_id]+=inter_hyper
                    inter_id+=1
    return communities,inter_hyperedge,hyperedges,map_T_id,num_nodes


def visualize_directed_factor_graph(communities: dict, hyperedges: list, num_nodes: int, inter_hyperedge_starts: list,map_T_id):
    """
    Visualize the directed hypergraph as a factor-graph-like layout:

        - nodes: ROIs, clustered by functional module on a circle
        - hyperedges: small square nodes, connected to incident ROIs
        - inter-module hyperedges: colored differently (by type)
    """
    # --------------------------------------------------------------
    # 1) Assign colors to nodes by module
    # --------------------------------------------------------------
    node_colors = [""] * num_nodes
    cluster_color_map = {name: plt.get_cmap("tab10").colors[i] for i, name in enumerate(communities.keys())}
    for name, node_ids in communities.items():
        for node_id in node_ids:
            if map_T_id[node_id] < num_nodes: node_colors[map_T_id[node_id]] = cluster_color_map[name]
    # Place nodes on circles around module centers
    pos = {}
    num_clusters = len(communities)
    for i, (name, node_ids) in enumerate(communities.items()):
        center_x = 10 * np.cos(i * 2 * np.pi / num_clusters)
        center_y = 10 * np.sin(i * 2 * np.pi / num_clusters)
        for node_id in node_ids:
            pos[map_T_id[node_id]] = np.array([center_x, center_y]) + np.random.rand(2) * 2.5 - 1.25

    # --------------------------------------------------------------
    # 2) Assign colors to inter-module hyperedges (flat ID list)
    # --------------------------------------------------------------
    inter_edge_color_map = {id: plt.get_cmap("tab20").colors[i] for i, nodes in enumerate(inter_hyperedge_starts) for id in nodes}

    # Flatten all inter-module hyperedge IDs into a set for membership check
    ranges = []
    for nodes in inter_hyperedge_starts:
        ranges+=nodes
    def get_edge_style(edge_id):
        """
        Return (color, linestyle, linewidth, label) for a given hyperedge ID.
        Inter-module edges: colored, dashed; intra-module: grey or ignored.
        """
        if edge_id in ranges:
            color = inter_edge_color_map[edge_id]
            label = f'intre-hyper-{i + 1}'
            return color, '--', 1.5, label
        else:
            return 'gray', '-', 0.0, 'intra-hyper'
    # --------------------------------------------------------------
    # 3) Draw nodes and hyperedge-centroid "factors"
    # --------------------------------------------------------------
    plt.figure(figsize=(8, 8)); ax = plt.gca()
    plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
    # Draw hyperedge factors and connections to nodes
    for edge in hyperedges:
        edge_id = edge['id']
        all_nodes = edge.get('tail', []) + edge.get('head', [])
        if not all_nodes: continue

        # Hyperedge "factor" placed at centroid of incident nodes
        centroid = np.mean([pos[map_T_id[node]] for node in all_nodes], axis=0)
        edge_color, linestyle, linewidth, _ = get_edge_style(edge_id)
        # Skip intra-module hyperedges if linewidth == 0
        if linewidth==0:
            continue
        # Plot hyperedge factor node
        ax.scatter(centroid[0], centroid[1], s=40, c=edge_color, marker='s')
        # Connect factor to each incident node
        for node in all_nodes:
            ax.plot([pos[map_T_id[node]][0], centroid[0]], [pos[map_T_id[node]][1], centroid[1]],
                    color=edge_color, linestyle=linestyle, linewidth=linewidth)
    # Draw ROI nodes
    ax.scatter([pos[i][0] for i in range(num_nodes)], [pos[i][1] for i in range(num_nodes)],
               c=node_colors, s=300, zorder=3, alpha=0.9, edgecolors='black')

    plt.title("Factor graph visualization of directed hypergraphs (colored by inter-cluster type)", fontsize=24)
    plt.axis('off')
    # plt.savefig("colored_inter_cluster_graph_flat_list.png", dpi=300, bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    # --------------------------------------------------------------
    # Run the full pipeline: build hypergraph, construct B, save .npz
    # --------------------------------------------------------------
    communities,inter_hyperedge,hyperedges,map_T_id,num_nodes = download_yeo_7_network_data()

    # visualize_directed_factor_graph(communities, hyperedges, num_nodes,inter_hyperedge,map_T_id)
    # Build incidence matrix B: (N compressed nodes × M hyperedges)
    M = len(hyperedges)
    N=num_nodes
    B = np.zeros((N,M))
    for l, hyper_edge in enumerate(hyperedges):
        # print(hyper_edge)
        tail_len = len(hyper_edge['tail'])
        head_len = len(hyper_edge['head'])
        for t in hyper_edge['tail']:
            B[map_T_id[t], l] = -1 / tail_len
        for h in hyper_edge['head']:
            B[map_T_id[h], l] = 1 / head_len
    # Turn list-of-lists into dict for convenience: index -> IDs
    inter_hyperedge = {i:value for i ,value in enumerate(inter_hyperedge)}
    # Save directed hypergraph data
    name = f"split_repaired_hypergraph_N{B.shape[0]}_M{B.shape[1]}"
    np.savez(
        name + '.npz',
        B=B,
        hyper_edges=np.array(hyperedges, dtype=object),
        Community=communities,
        inter_hyperedge=inter_hyperedge,
        map_T_id = map_T_id
    )

