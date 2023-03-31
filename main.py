import os
import json

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from sentence_transformers import util

from model import Encoder, run_training
from data_handling import DataLoader
from romeV4 import RomeV4Handler
from pole_emploi_API import Scrapper

from utils import (
    scrap_jobs_pole_emploi,
    sumup_scrapping_pole_emploi,
    precompute_job_descriptions,
    precompute_ROMEv4_embeddings,
    precompute_ROMEv4_category_embeddings
)

LOCAL_DATA_PATH = "/mnt/data/Noese/"

if __name__ == "__main__":

    """encoder = Encoder(device=torch.device("cpu"))

    data_loader = DataLoader(data_path=LOCAL_DATA_PATH)
    #precompute_job_descriptions(encoder, data_loader, LOCAL_DATA_PATH)

    romev4_handler = RomeV4Handler(data_path=LOCAL_DATA_PATH)
    #precompute_ROMEv4_embeddings(encoder, romev4_handler, LOCAL_DATA_PATH)
    #precompute_ROMEv4_category_embeddings(encoder, romev4_handler, LOCAL_DATA_PATH)

    #scrapper_pole_emploi = Scrapper()

    #scrap_jobs_pole_emploi(romev4_handler, scrapper_pole_emploi, LOCAL_DATA_PATH)
    #sumup_scrapping_pole_emploi(LOCAL_DATA_PATH)

    embs_job_descriptions = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_descriptions.npy")
    embs_job_profiles = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_profiles.npy")
    embs_job_titles = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_titles.npy")

    embs_romev4_jobs_title = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_romev4.npy")
    #embs_romev4_jobs_category = np.load(LOCAL_DATA_PATH + "precomputed/embeddings_romev4_category.npy")"""

    run_training(precomputed_embeddings=True)
