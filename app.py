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
        except:
            return None
    
    # Assuming Column A is 'Product' and the rest are descriptors
    product_col = df.columns[0]
    df.set_index(product_col, inplace=True)
    df.fillna(0, inplace=True)
    return df

# --- Sidebar / Data Upload ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload ODNA Excel File", type=["xls", "xlsx"])

df = load_data(uploaded_file)

if df is not None:
    st.sidebar.success(f"Loaded {df.shape[0]} fragrances and {df.shape[1]} descriptors.")
    
    # Standardize/Normalize data
    from sklearn.preprocessing import StandardScaler
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
        method = st.selectbox("Select Dimensionality Reduction Method:", ["PCA", "t-SNE"])
        
        if method == "PCA":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            
        coords = reducer.fit_transform(data_scaled)
        viz_df = pd.DataFrame(coords, columns=["Dim 1", "Dim 2"], index=df.index).reset_index()
        
        fig = px.scatter(viz_df, x="Dim 1", y="Dim 2", hover_name=df.index.name,
                         title=f"2D Landscape using {method}")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: Clusters & Heroes ---
    with tab2:
        st.header("2. Objective Clustering & Heroes")
        n_clusters = st.slider("Number of Clusters (Families):", min_value=2, max_value=20, value=8)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Identify Heroes (closest to centroid)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data_scaled)
        heroes = df.index[closest].tolist()
        
        cluster_df = pd.DataFrame({"Cluster": clusters}, index=df.index)
        
        st.write("### Cluster Representatives (Heroes)")
        hero_display = pd.DataFrame({"Cluster": range(n_clusters), "Hero Fragrance": heroes})
        st.dataframe(hero_display, hide_index=True)

        # Plot clusters on the same 2D landscape
        viz_df["Cluster"] = clusters.astype(str)
        fig_clusters = px.scatter(viz_df, x="Dim 1", y="Dim 2", color="Cluster", hover_name=df.index.name,
                                  title=f"Fragrance Clusters (k={n_clusters})")
        
        # Add stars for heroes
        hero_coords = viz_df[viz_df[df.index.name].isin(heroes)]
        fig_clusters.add_trace(go.Scatter(x=hero_coords["Dim 1"], y=hero_coords["Dim 2"],
                                          mode='markers', marker=dict(symbol='star', size=15, color='black'),
                                          name='Heroes', hovertext=hero_coords[df.index.name]))
        
        st.plotly_chart(fig_clusters, use_container_width=True)

    # --- TAB 3: Network Map (Substitution Pathways) ---
    with tab3:
        st.header("3. Fragrance Substitution Network Map")
        st.markdown("This map reveals continuous substitution pathways. Fragrances connected by a line are highly similar.")
        
        threshold = st.slider("Similarity Threshold (0.0 to 1.0):", min_value=0.5, max_value=0.99, value=0.85, step=0.01,
                              help="Higher values mean fewer connections (stricter similarity requirement).")
        
        # Build NetworkX Graph
        G = nx.Graph()
        
        # Add nodes
        for node in df.index:
            G.add_node(node)
            
        # Add edges based on threshold
        for i in range(len(df.index)):
            for j in range(i+1, len(df.index)):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(df.index[i], df.index[j], weight=sim)
                    
        # Calculate layout
        pos = nx.spring_layout(G, seed=42)
        
        # Plotly Network edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        # Plotly Network nodes
        node_x = []
        node_y = []
        node_text = []
        node_adjacencies = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_adjacencies.append(len(list(G.neighbors(node))))
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True,
                                            color=node_adjacencies, size=10, 
                                            colorbar=dict(thickness=15, title='Number of Connections', xanchor='left', titleside='right')))
        node_trace.text = node_text
        
        fig_net = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        st.plotly_chart(fig_net, use_container_width=True)

    # --- TAB 4: Similarity Engine ---
    with tab4:
        st.header("4. Fragrance Similarity Engine")
        st.markdown("Select a fragrance to find its closest olfactive alternatives.")
        
        target = st.selectbox("Select Target Fragrance:", df.index)
        top_n = st.number_input("Number of Alternatives to show:", min_value=1, max_value=20, value=5)
        
        if st.button("Find Alternatives"):
            # Get target similarities, drop the target itself, and sort
            target_sims = sim_df[target].drop(target).sort_values(ascending=False)
            
            top_alternatives = target_sims.head(top_n).reset_index()
            top_alternatives.columns = ["Alternative Fragrance", "Proximity Score (0 to 1)"]
            
            # Format score as a percentage for readability
            top_alternatives["Proximity Match"] = (top_alternatives["Proximity Score (0 to 1)"] * 100).round(2).astype(str) + "%"
            
            st.dataframe(top_alternatives[["Alternative Fragrance", "Proximity Match"]], hide_index=True)
            
            # Optional: Show the olfactive profile comparison chart
            comp_df = df.loc[[target] + top_alternatives["Alternative Fragrance"].tolist()].T
            comp_df = comp_df.loc[(comp_df != 0).any(axis=1)] # drop empty rows for cleaner chart
            
            st.write("### Olfactive Profile Comparison")
            st.line_chart(comp_df)

else:
    st.info("Please upload the 'Example ODNA.xls' file in the sidebar to begin.")
