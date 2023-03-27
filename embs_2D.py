import numpy as np

from umap import UMAP

import streamlit as st
import plotly.express as px

from data_loader import DataLoader

def reduce_dimensions(embeddings):
    reducer = UMAP(
        n_components = 2,
        n_neighbors = 15,
        metric = "cosine",
        init = "spectral",
        spread = 1.0,
        low_memory = True,
        unique = True,
        verbose = True,
        n_jobs = 8,
        transform_seed = 42,
        random_state = 1337
    )

    reduced = reducer.fit_transform(embeddings)

    return reduced

def populate_reduced_dim(reduced, df):

    for i in range(reduced.shape[1]):
        df[f"Dimension {i}"] = reduced[:, i]

    return df

LOCAL_DATA_PATH = "/mnt/data/Noese/"

embs_job_descriptions = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_descriptions_1k.npy")
embs_job_profiles = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_profiles_1k.npy")
embs_job_titles = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_titles_1k.npy")

embs_mean = np.mean(
    (
        embs_job_descriptions,
        embs_job_profiles,
        embs_job_titles
    ),
    axis=0
)

embs_mean_2D = reduce_dimensions(embs_mean)

data_loader = DataLoader(data_path=LOCAL_DATA_PATH)
df = data_loader.df_job[:1000]

df = populate_reduced_dim(embs_mean_2D, df)

fig = px.scatter(
        df,
        x = "Dimension 0",
        y = "Dimension 1",
        #color = "cluster",
        #size = "_size",
        size_max = 25,
        hover_name = "title",
        hover_data = [
            "description",
            "profil"
        ],
        title = "TEST"
    )
fig.update_layout(autosize=False, height=800, width=800)
st.plotly_chart(fig, use_container_width=True)
