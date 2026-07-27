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

# --- Helper & Data Loading Functions ---
@st.cache_data
def load_data(uploaded_file):
    """
    Loads input Excel workbook with 2 tabs: Characterizer & Family.
    Column A: Random Code
    Column B: Group Code
    Column C: Brand Name
    Column D: Fantasy Name
    Columns E to CN: Characterizer descriptors for mathematical modeling.
    """
    if uploaded_file is not None:
        file_source = uploaded_file
    else:
        # Fallback local files
        try:
            file_source = "ODNA VICTOR PORTFOLIO.xlsx"
            xls = pd.ExcelFile(file_source)
        except Exception:
            try:
                file_source = "Example ODNA.xls"
                xls = pd.ExcelFile(file_source)
            except Exception:
                return None, None, None, None, None, None

    xls = pd.ExcelFile(file_source)
    sheet_names = xls.sheet_names

    # Identify sheets based on name patterns
    char_sheet, fam_sheet = None, None
    for name in sheet_names:
        lname = name.lower()
        if "char" in lname:
            char_sheet = name
        elif "fam" in lname:
            fam_sheet = name

    if char_sheet is None:
        char_sheet = sheet_names[0]
    if fam_sheet is None and len(sheet_names) > 1:
        fam_sheet = sheet_names[1]

    # 1. Read Characterizers Sheet
    df_char = pd.read_excel(xls, sheet_name=char_sheet)
    
    col_a_code = df_char.columns[0]      # Random Code
    col_b_group = df_char.columns[1]     # Group Code
    col_c_brand = df_char.columns[2]     # Brand Name
    col_d_fantasy = df_char.columns[3]   # Fantasy Name
    
    # Extract Characterizer features from Column E (index 4) onwards
    feature_cols = df_char.columns[4:].tolist()
    
    # Feature Dataframe (Numerical matrix for analysis)
    df_feat = df_char.set_index(col_a_code)[feature_cols].fillna(0)
    
    # Metadata Dataframe
    df_meta = df_char[[col_a_code, col_b_group, col_c_brand, col_d_fantasy]].copy()
    df_meta.set_index(col_a_code, inplace=True)

    # 2. Compute ODNA Family from Family Sheet
    if fam_sheet and fam_sheet in xls.sheet_names:
        df_fam = pd.read_excel(xls, sheet_name=fam_sheet)
        fam_code_col = df_fam.columns[0]
        fam_desc_cols = df_fam.columns[4:]  # Family columns starting at Col E
        
        fam_matrix = df_fam.set_index(fam_code_col)[fam_desc_cols].fillna(0)
        
        # Row-wise argmax to get the main family
        has_family = fam_matrix.sum(axis=1) > 0
        odna_series = pd.Series("Unclassified", index=fam_matrix.index)
        odna_series[has_family] = fam_matrix[has_family].idxmax(axis=1)
        
        df_meta["ODNA Family"] = df_meta.index.map(odna_series).fillna("Unclassified")
    else:
        df_meta["ODNA Family"] = "Unclassified"

    return df_feat, df_meta, col_a_code, col_b_group, col_c_brand, col_d_fantasy

def get_top_notes_str(row, top_k=3):
    """Extract top_k non-zero characterizer descriptors for hover text."""
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
uploaded_file = st.sidebar.file_uploader("Upload ODNA Excel File (Multi-Tab)", type=["xls", "xlsx"])

df_feat, df_meta, col_a, col_b, col_c, col_d = load_data(uploaded_file)

if df_feat is not None:
    st.sidebar.success(f"Loaded {df_feat.shape[0]} fragrances with {df_feat.shape[1]} characterizers.")
    
    # Top notes summary for each fragrance
    top_notes_series = df_feat.apply(get_top_notes_str, axis=1)
    
    # Standardize data for PCA/t-SNE/KMeans
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_feat)
    
    # Cosine Similarity Matrix
    sim_matrix = cosine_similarity(df_feat)
    sim_df = pd.DataFrame(sim_matrix, index=df_feat.index, columns=df_feat.index)

    # --- UI TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Olfactive Landscape", 
        "Clusters & Heroes", 
        "Network Map", 
        "Similarity Engine"
    ])

    # =========================================================================
    # --- TAB 1: Visualizing the Olfactive Landscape ---
    # =========================================================================
    with tab1:
        st.header("1. Visualizing the Olfactive Landscape")
        st.markdown("Explore fragrance positioning in 2D space. Map display strictly showcases **Random Codes**.")
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            method = st.selectbox("Select Dimensionality Reduction Method:", ["PCA", "t-SNE"])
        with col_m2:
            show_density = st.checkbox("🔥 Overlay Territory Density Heatmap (White Space Detector)", value=True)
        
        if method == "PCA":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(data_scaled)
            pc1_drivers = pd.Series(reducer.components_[0], index=df_feat.columns).abs().nlargest(4)
            pc2_drivers = pd.Series(reducer.components_[1], index=df_feat.columns).abs().nlargest(4)
        else:
            safe_perplexity = min(30, max(1, len(df_feat) - 1))
            reducer = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42)
            coords = reducer.fit_transform(data_scaled)
            
        viz_df = pd.DataFrame(coords, columns=["Dim 1", "Dim 2"], index=df_feat.index).reset_index()
        viz_df["Top Notes"] = top_notes_series.values
        viz_df["Fantasy Name"] = viz_df[col_a].map(df_meta[col_d])
        viz_df["Brand Name"] = viz_df[col_a].map(df_meta[col_c])
        viz_df["ODNA Family"] = viz_df[col_a].map(df_meta["ODNA Family"])
        
        if show_density:
            fig = go.Figure()
            
            # Density Contour Layer
            fig.add_trace(go.Histogram2dContour(
                x=viz_df["Dim 1"],
                y=viz_df["Dim 2"],
                colorscale="YlOrRd",
                reversescale=False,
                showscale=True,
                name="Density",
                contours=dict(coloring='heatmap', showlines=True),
                opacity=0.6,
                colorbar=dict(title="SKU Density")
            ))
            
            # Hover text details
            hover_text = [
                f"<b>Code: {row[col_a]}</b><br>Name: {row['Fantasy Name']}<br>Brand: {row['Brand Name']}<br>Family: {row['ODNA Family']}<br>Top Notes: {row['Top Notes']}" 
                for _, row in viz_df.iterrows()
            ]
            
            # Show ONLY Random Codes on Map Labels
            fig.add_trace(go.Scatter(
                x=viz_df["Dim 1"],
                y=viz_df["Dim 2"],
                mode='markers+text',
                text=viz_df[col_a],  # ONLY Random Code displayed
                textposition="top center",
                marker=dict(size=9, color='black', line=dict(width=1, color='white')),
                hoverinfo='text',
                hovertext=hover_text,
                name='Fragrances'
            ))
            
            fig.update_layout(
                title=f"2D Olfactive Territory Density Heatmap ({method})",
                xaxis_title="Dim 1", yaxis_title="Dim 2",
                hovermode='closest'
            )
        else:
            fig = px.scatter(
                viz_df, x="Dim 1", y="Dim 2", text=col_a,
                hover_data={"Dim 1": ":.2f", "Dim 2": ":.2f", "Fantasy Name": True, "Brand Name": True, "ODNA Family": True, "Top Notes": True},
                title=f"2D Olfactive Landscape using {method}"
            )
            fig.update_traces(marker=dict(size=9, opacity=0.8), textposition='top center')

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 How to Read the Territory Density Heatmap & Spot 'White Spaces'"):
            st.markdown("""
            * **🔥 Dark / Warm Density Hubs:** High concentration of existing SKUs. These zones present **cannibalization risks** and are prime candidates for portfolio consolidation.
            * **⚪ Pale / Low-Density Zones:** **"White Spaces"** representing unserved olfactive profiles. These zones highlight **New Product Development (NPD)** launch opportunities.
            """)

        if method == "PCA":
            with st.expander("🔍 What is driving these axes? (PCA Feature Importance)"):
                col_a_p, col_b_p = st.columns(2)
                with col_a_p:
                    st.write("**Top Drivers for Dimension 1 (Horizontal):**")
                    st.write(", ".join(pc1_drivers.index))
                with col_b_p:
                    st.write("**Top Drivers for Dimension 2 (Vertical):**")
                    st.write(", ".join(pc2_drivers.index))

    # =========================================================================
    # --- TAB 2: Objective Clustering & Olfactive Signatures (Option A: Barycenter/Medoid) ---
    # =========================================================================
    with tab2:
        st.header("2. Objective Clustering & Olfactive Signatures")
        st.markdown("Group fragrances into olfactive families using **Option A: Cluster Barycenter / Medoid** selection.")
        
        max_clusters = min(20, len(df_feat))
        n_clusters = st.slider("Number of Clusters (Families):", min_value=2, max_value=max_clusters, value=min(8, max_clusters))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        # --- Option A: Strictly Find Intra-Cluster Medoid (Closest SKU to Centroid within Cluster) ---
        heroes_codes = []
        hero_distances = []
        
        for c_id in range(n_clusters):
            cluster_mask = (clusters == c_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                cluster_data = data_scaled[cluster_indices]
                centroid = kmeans.cluster_centers_[c_id].reshape(1, -1)
                
                # Minimum distance inside this specific cluster
                closest_idx_within, min_dist = pairwise_distances_argmin_min(centroid, cluster_data)
                hero_global_idx = cluster_indices[closest_idx_within[0]]
                
                heroes_codes.append(df_feat.index[hero_global_idx])
                hero_distances.append(min_dist[0])
            else:
                heroes_codes.append(None)
                hero_distances.append(0.0)
        
        # Compute cluster average profiles
        df_clustered = df_feat.copy()
        df_clustered["Cluster"] = clusters
        cluster_means = df_clustered.groupby("Cluster").mean()
        
        # Build Cluster Summary Table
        cluster_summary = []
        for c_id in range(n_clusters):
            c_size = (clusters == c_id).sum()
            hero_code = heroes_codes[c_id]
            hero_meta = df_meta.loc[hero_code]
            dist_val = hero_distances[c_id]
            
            top_c_notes = cluster_means.loc[c_id].nlargest(4)
            sig_str = ", ".join([f"{k} ({v:.1f})" for k, v in top_c_notes.items() if v > 0])
            
            cluster_summary.append({
                "Cluster ID": c_id,
                "Size": c_size,
                "Hero Medoid (A)": hero_code,
                "Fantasy Name (D)": hero_meta[col_d],
                "Group Code (B)": hero_meta[col_b],
                "Brand Name (C)": hero_meta[col_c],
                "ODNA Family": hero_meta["ODNA Family"],
                "Barycenter Dist.": f"{dist_val:.2f}",
                "Olfactive Signature (Dominant Notes)": sig_str
            })
            
        summary_df = pd.DataFrame(cluster_summary)
        st.write("### Cluster Signatures & Barycenter Medoids (Heroes)")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # Plot clusters on 2D landscape
        viz_df["Cluster"] = clusters.astype(str)
        fig_clusters = px.scatter(
            viz_df, x="Dim 1", y="Dim 2", color="Cluster", text=col_a,
            hover_data={"Fantasy Name": True, "Brand Name": True, "ODNA Family": True, "Top Notes": True, "Cluster": True, "Dim 1": False, "Dim 2": False},
            title=f"Fragrance Territory Clusters & Barycenter Heroes (k={n_clusters})"
        )
        fig_clusters.update_traces(marker=dict(size=9), textposition="top center")
        
        # Add stars for Medoid Heroes (Labelled strictly with Random Code)
        hero_coords = viz_df[viz_df[col_a].isin(heroes_codes)]
        fig_clusters.add_trace(go.Scatter(
            x=hero_coords["Dim 1"], y=hero_coords["Dim 2"],
            mode='markers+text',
            marker=dict(symbol='star', size=16, color='black', line=dict(width=1, color='white')),
            text=hero_coords[col_a],
            textposition="top center",
            name='Cluster Medoids (Heroes)',
            hovertext=hero_coords["Fantasy Name"] + " (" + hero_coords["ODNA Family"] + ")"
        ))
        
        st.plotly_chart(fig_clusters, use_container_width=True)

        # Deep Dive: Cluster Olfactive Fingerprint
        st.subheader("📊 Deep Dive: Cluster Olfactive Fingerprint")
        selected_c = st.selectbox("Select a Cluster to inspect:", range(n_clusters), format_func=lambda x: f"Cluster {x} (Hero: {heroes_codes[x]})")
        
        c_col1, c_col2 = st.columns([1, 1])
        
        with c_col1:
            top_6_notes = cluster_means.loc[selected_c].nlargest(6)
            overall_means = df_feat.mean()[top_6_notes.index]
            
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
            member_codes = df_clustered[df_clustered["Cluster"] == selected_c].index.tolist()
            
            member_details = []
            for code in member_codes:
                meta = df_meta.loc[code]
                member_details.append({
                    "Random Code (A)": code,
                    "Fantasy Name (D)": meta[col_d],
                    "Group Code (B)": meta[col_b],
                    "Brand Name (C)": meta[col_c],
                    "ODNA Family": meta["ODNA Family"],
                    "Is Hero Medoid": "⭐ Yes" if code == heroes_codes[selected_c] else "No",
                    "Top Notes": top_notes_series[code]
                })
            st.dataframe(pd.DataFrame(member_details), hide_index=True, use_container_width=True)

    # =========================================================================
    # --- TAB 3: Fragrance Substitution Network Map ---
    # =========================================================================
    with tab3:
        st.header("3. Fragrance Substitution Network Map")
        st.markdown("Reveals continuous substitution pathways. Nodes display strictly **Random Codes**.")
        
        threshold = st.slider("Similarity Threshold (0.0 to 1.0):", min_value=0.5, max_value=0.99, value=0.85, step=0.01,
                              help="Higher values mean fewer connections (stricter similarity requirement).")
        
        G = nx.Graph()
        for node in df_feat.index:
            G.add_node(node)
            
        for i in range(len(df_feat.index)):
            for j in range(i+1, len(df_feat.index)):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(df_feat.index[i], df_feat.index[j], weight=sim)
                    
        pos = nx.spring_layout(G, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        node_x, node_y, node_hover, node_adjacencies = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            n_conn = len(list(G.neighbors(node)))
            node_adjacencies.append(n_conn)
            
            meta = df_meta.loc[node]
            node_hover.append(
                f"<b>{node}</b><br>Fantasy: {meta[col_d]}<br>Brand: {meta[col_c]}<br>Family: {meta['ODNA Family']}<br>Connections: {n_conn}"
            )
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo='text', hovertext=node_hover,
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

    # =========================================================================
    # --- TAB 4: Similarity Engine ---
    # =========================================================================
    with tab4:
        st.header("4. Fragrance Similarity Engine")
        st.markdown("Select a target fragrance to locate its closest alternatives and evaluate substitution feasibility.")
        
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
            
        target = st.selectbox("Select Target Fragrance (Random Code):", df_feat.index, 
                            format_func=lambda x: f"{x} - {df_meta.loc[x, col_d]} ({df_meta.loc[x, col_c]})")
        top_n = st.number_input("Number of Alternatives to show:", min_value=1, max_value=20, value=5)
        
        if st.button("Find Alternatives"):
            target_sims = sim_df[target].drop(target).sort_values(ascending=False)
            top_alternatives = target_sims.head(top_n).reset_index()
            top_alternatives.columns = ["Random Code (A)", "Proximity Score"]
            
            top_alternatives["Fantasy Name (D)"] = top_alternatives["Random Code (A)"].map(df_meta[col_d])
            top_alternatives["Group Code (B)"] = top_alternatives["Random Code (A)"].map(df_meta[col_b])
            top_alternatives["Brand Name (C)"] = top_alternatives["Random Code (A)"].map(df_meta[col_c])
            top_alternatives["ODNA Family"] = top_alternatives["Random Code (A)"].map(df_meta["ODNA Family"])
            
            top_alternatives["Tier Status"] = top_alternatives["Proximity Score"].apply(classify_proximity)
            top_alternatives["Proximity Match"] = (top_alternatives["Proximity Score"] * 100).round(1).astype(str) + "%"
            top_alternatives["Top Notes"] = top_alternatives["Random Code (A)"].map(top_notes_series)
            
            output_cols = [
                "Random Code (A)", "Fantasy Name (D)", "Group Code (B)", 
                "Brand Name (C)", "ODNA Family", "Tier Status", "Proximity Match", "Top Notes"
            ]
            
            st.dataframe(
                top_alternatives[output_cols], 
                hide_index=True, 
                use_container_width=True
            )
            
            comp_codes = [target] + top_alternatives["Random Code (A)"].tolist()
            comp_df = df_feat.loc[comp_codes].T
            comp_df = comp_df.loc[(comp_df != 0).any(axis=1)]
            
            st.write("### Olfactive Characterizer Profile Comparison")
            st.line_chart(comp_df)

else:
    st.info("Please upload your workbook (containing 'characterizer' and 'Family' tabs) in the sidebar to begin.")
