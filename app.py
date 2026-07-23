import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="Fragrance Portfolio Rationalizer", layout="wide")
st.title("🌸 Fragrance Portfolio Rationalization & Network Analysis")

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        # Fallback to local file if it exists in the directory
        try:
            df = pd.read_excel("Example ODNA.xls")
        except Exception:
            return None
    
    # Assuming Column A is 'Product' and the rest are descriptors
    product_col = df.columns[0]
    df.set_index(product_col, inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_top_notes_str(row, top_k=3):
    """Extract top_k non-zero descriptors for hover text."""
    top = row[row > 0].nlargest(top_k)
    if len(top) == 0:
        return "None"
    return ", ".join([f"{note}: {val:.1f}" for note, val in top.items()])

def classify_proximity(score):
    """Categorizes raw similarity score into strategic portfolio action tiers."""
    if score >= 0.80:
        return "🟩 Direct Substitute"
    elif score >= 0.65:
        return "🟦 Strong Alternative"
    elif score >= 0.50:
        return "🟨 Sub-Family Neighbor"
    else:
        return "⬜ Distinct Territory"

# --- Sidebar / Data Upload ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload ODNA Excel File", type=["xls", "xlsx"])

df = load_data(uploaded_file)

if df is not None:
    st.sidebar.success(f"Loaded {df.shape[0]} fragrances and {df.shape[1]} descriptors.")
    
    # Compute Top 3 dominant notes for every fragrance for rich tooltips
    top_notes_series = df.apply(get_top_notes_str, axis=1)
    
    # Standardize/Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    # Similarity Matrix (Cosine)
    sim_matrix = cosine_similarity(df)
    sim_df = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)

    # --- UI TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Olfactive Landscape", 
        "Clusters & Heroes", 
        "Network Map", 
        "Similarity Engine"
    ])

    # --- TAB 1: Olfactive Landscape ---
    with tab1:
        st.header("1. Visualizing the Olfactive Landscape")
        st.markdown("Explore how fragrances are positioned in 2D space based on their olfactive profiles.")
        
        method = st.selectbox("Select Dimensionality Reduction Method:", ["PCA", "t-SNE"])
        
        if method == "PCA":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(data_scaled)
            
            # Feature loadings to explain PCA axes
            pc1_drivers = pd.Series(reducer.components_[0], index=df.columns).abs().nlargest(4)
            pc2_drivers = pd.Series(reducer.components_[1], index=df.columns).abs().nlargest(4)
        else:
            safe_perplexity = min(30, max(1, len(df) - 1))
            reducer = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42)
            coords = reducer.fit_transform(data_scaled)
            
        viz_df = pd.DataFrame(coords, columns=["Dim 1", "Dim 2"], index=df.index).reset_index()
        viz_df["Top Notes"] = top_notes_series.values
        
        fig = px.scatter(
            viz_df, x="Dim 1", y="Dim 2", hover_name=df.index.name,
            hover_data={"Top Notes": True, "Dim 1": ":.2f", "Dim 2": ":.2f"},
            title=f"2D Landscape using {method}"
        )
        fig.update_traces(marker=dict(size=9, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

        if method == "PCA":
            with st.expander("🔍 What is driving these axes? (PCA Feature Importance)"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Top Drivers for Dimension 1 (Horizontal):**")
                    st.write(", ".join(pc1_drivers.index))
                with col_b:
                    st.write("**Top Drivers for Dimension 2 (Vertical):**")
                    st.write(", ".join(pc2_drivers.index))

    # --- TAB 2: Clusters & Heroes ---
    with tab2:
        st.header("2. Objective Clustering & Olfactive Signatures")
        st.markdown("Group fragrances into olfactive families, identify their dominant notes, and select cluster heroes.")
        
        max_clusters = min(20, len(df))
        n_clusters = st.slider("Number of Clusters (Families):", min_value=2, max_value=max_clusters, value=min(8, max_clusters))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Identify Heroes (closest product to centroid)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data_scaled)
        heroes = df.index[closest].tolist()
        
        # Compute cluster average profiles
        df_clustered = df.copy()
        df_clustered["Cluster"] = clusters
        cluster_means = df_clustered.groupby("Cluster").mean()
        
        # Build Rich Cluster Summary Table
        cluster_summary = []
        for c_id in range(n_clusters):
            c_size = (clusters == c_id).sum()
            c_hero = heroes[c_id]
            # Get top 4 dominant notes in cluster
            top_c_notes = cluster_means.loc[c_id].nlargest(4)
            sig_str = ", ".join([f"{k} ({v:.1f})" for k, v in top_c_notes.items() if v > 0])
            
            cluster_summary.append({
                "Cluster ID": c_id,
                "Size": c_size,
                "Hero Fragrance": c_hero,
                "Olfactive Signature (Dominant Notes)": sig_str
            })
            
        summary_df = pd.DataFrame(cluster_summary)
        
        st.write("### Cluster Signatures & Representative Heroes")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # Plot clusters on the 2D landscape
        viz_df["Cluster"] = clusters.astype(str)
        fig_clusters = px.scatter(
            viz_df, x="Dim 1", y="Dim 2", color="Cluster", hover_name=df.index.name,
            hover_data={"Top Notes": True, "Cluster": True, "Dim 1": False, "Dim 2": False},
            title=f"Fragrance Territory Clusters (k={n_clusters})"
        )
        fig_clusters.update_traces(marker=dict(size=9))
        
        # Add stars for heroes
        hero_coords = viz_df[viz_df[df.index.name].isin(heroes)]
        fig_clusters.add_trace(go.Scatter(
            x=hero_coords["Dim 1"], y=hero_coords["Dim 2"],
            mode='markers+text',
            marker=dict(symbol='star', size=16, color='black', line=dict(width=1, color='white')),
            text=hero_coords[df.index.name],
            textposition="top center",
            name='Heroes',
            hovertext=hero_coords["Top Notes"]
        ))
        
        st.plotly_chart(fig_clusters, use_container_width=True)

        # --- Interactive Cluster Inspector ---
        st.subheader("📊 Deep Dive: Cluster Olfactive Fingerprint")
        selected_c = st.selectbox("Select a Cluster to inspect:", range(n_clusters), format_func=lambda x: f"Cluster {x} (Hero: {heroes[x]})")
        
        c_col1, c_col2 = st.columns([1, 1])
        
        with c_col1:
            # Radar chart of top 6 notes for this cluster vs overall average
            top_6_notes = cluster_means.loc[selected_c].nlargest(6)
            overall_means = df.mean()[top_6_notes.index]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top_6_notes.values, theta=top_6_notes.index, fill='toself', name=f'Cluster {selected_c}'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=overall_means.values, theta=overall_means.index, fill='toself', name='Portfolio Average', opacity=0.5
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=f"Olfactive Fingerprint: Cluster {selected_c}"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with c_col2:
            st.write(f"**Fragrances in Cluster {selected_c}:**")
            member_fragrances = df_clustered[df_clustered["Cluster"] == selected_c].index.tolist()
            
            member_details = []
            for frag in member_fragrances:
                member_details.append({
                    "Fragrance": frag,
                    "Is Hero": "⭐ Yes" if frag == heroes[selected_c] else "No",
                    "Top Notes": top_notes_series[frag]
                })
            st.dataframe(pd.DataFrame(member_details), hide_index=True, use_container_width=True)

    # --- TAB 3: Network Map (Substitution Pathways) ---
    with tab3:
        st.header("3. Fragrance Substitution Network Map")
        st.markdown("This map reveals continuous substitution pathways. Fragrances connected by a line are highly similar.")
        
        threshold = st.slider("Similarity Threshold (0.0 to 1.0):", min_value=0.5, max_value=0.99, value=0.85, step=0.01,
                              help="Higher values mean fewer connections (stricter similarity requirement).")
        
        # Build NetworkX Graph
        G = nx.Graph()
        
        for node in df.index:
            G.add_node(node)
            
        for i in range(len(df.index)):
            for j in range(i+1, len(df.index)):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(df.index[i], df.index[j], weight=sim)
                    
        pos = nx.spring_layout(G, seed=42)
        
        # Edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        # Nodes
        node_x, node_y, node_hover, node_adjacencies = [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            n_conn = len(list(G.neighbors(node)))
            node_adjacencies.append(n_conn)
            node_hover.append(f"<b>{node}</b><br>Connections: {n_conn}<br>Top Notes: {top_notes_series[node]}")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_hover,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_adjacencies,
                size=12, 
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Number of Connections', side='right'),
                    xanchor='left'
                )
            )
        )
        
        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        st.plotly_chart(fig_net, use_container_width=True)

    # --- TAB 4: Similarity Engine ---
    with tab4:
        st.header("4. Fragrance Similarity Engine")
        st.markdown("Select a target fragrance to locate its closest alternatives and evaluate substitution feasibility.")
        
        # Collapsible Tier Definition Reference Table
        with st.expander("ℹ️ How to Interpret Olfactive Proximity Tiers"):
            tier_guide_df = pd.DataFrame([
                {
                    "Tier Status": "🟩 Direct Substitute",
                    "Score Range": "≥ 80%",
                    "Match Level": "Near-Duplicate",
                    "Strategic Action & Interpretation": "Prime rationalization candidate. Shared dominant notes in near-identical proportions; high confidence for SKU consolidation or formula replacement."
                },
                {
                    "Tier Status": "🟦 Strong Alternative",
                    "Score Range": "65% – 79%",
                    "Match Level": "High Proximity",
                    "Strategic Action & Interpretation": "Line variant / Reformulation candidate. Shares core family signature; suitable substitute requiring minor adjustment."
                },
                {
                    "Tier Status": "🟨 Sub-Family Neighbor",
                    "Score Range": "50% – 64%",
                    "Match Level": "Moderate Match",
                    "Strategic Action & Interpretation": "Cross-category bridge. Shares overarching framework, but features noticeably distinct top or bottom notes."
                },
                {
                    "Tier Status": "⬜ Distinct Territory",
                    "Score Range": "< 50%",
                    "Match Level": "Low Proximity",
                    "Strategic Action & Interpretation": "Distant profile. Unique space in portfolio; minimal risk of cannibalization."
                }
            ])
            st.dataframe(tier_guide_df, hide_index=True, use_container_width=True)
            
        target = st.selectbox("Select Target Fragrance:", df.index)
        top_n = st.number_input("Number of Alternatives to show:", min_value=1, max_value=20, value=5)
        
        if st.button("Find Alternatives"):
            target_sims = sim_df[target].drop(target).sort_values(ascending=False)
            top_alternatives = target_sims.head(top_n).reset_index()
            top_alternatives.columns = ["Alternative Fragrance", "Proximity Score"]
            
            # Apply tier classification and formatting
            top_alternatives["Tier Status"] = top_alternatives["Proximity Score"].apply(classify_proximity)
            top_alternatives["Proximity Match"] = (top_alternatives["Proximity Score"] * 100).round(1).astype(str) + "%"
            top_alternatives["Top Notes"] = top_alternatives["Alternative Fragrance"].map(top_notes_series)
            
            st.dataframe(
                top_alternatives[["Alternative Fragrance", "Tier Status", "Proximity Match", "Top Notes"]], 
                hide_index=True, 
                use_container_width=True
            )
            
            # Olfactive profile comparison chart
            comp_df = df.loc[[target] + top_alternatives["Alternative Fragrance"].tolist()].T
            comp_df = comp_df.loc[(comp_df != 0).any(axis=1)] # Filter non-zero descriptors
            
            st.write("### Olfactive Profile Comparison")
            st.line_chart(comp_df)

else:
    st.info("Please upload the 'Example ODNA.xls' file in the sidebar to begin.")
