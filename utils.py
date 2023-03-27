import os
import base64

import json

from tqdm import tqdm

import numpy as np

def precompute_ROMEv4_embeddings(encoder, romev4_handler, path):
    
    texts = romev4_handler.get_jobs_title()
    embeddings = encoder.encode(sentences=texts)

    with open(path + "precomputed/embeddings_romev4.npy", 'wb') as f:
        np.save(f, embeddings)

def precompute_ROMEv4_category_embeddings(encoder, romev4_handler, path):
    
    texts = [
        tup[1]
        for tup
        in romev4_handler.get_jobs_category()
        ]
    embeddings = encoder.encode(sentences=texts)

    with open(path + "precomputed/embeddings_romev4_category.npy", 'wb') as f:
        np.save(f, embeddings)
    
def precompute_job_descriptions(encoder, data_loader, path):

    # Job Descriptions
    texts = data_loader.get_job_descriptions()
    embeddings = encoder.encode(sentences=texts)

    with open(path + "precomputed/embeddings_descriptions.npy", 'wb') as f:
        np.save(f, embeddings)
    
    # Job Profiles
    texts = data_loader.get_job_profiles()
    embeddings = encoder.encode(sentences=texts)

    with open(path + "precomputed/embeddings_profiles.npy", 'wb') as f:
        np.save(f, embeddings)

    # Job Titles
    texts = data_loader.get_job_titles()
    embeddings = encoder.encode(sentences=texts)

    with open(path + "precomputed/embeddings_titles.npy", 'wb') as f:
        np.save(f, embeddings)

def scrap_jobs_pole_emploi(romev4_handler, scrapper_pole_emploi, path):
    """
    Scrap Job Offers from all categories (+11k) through Pole Emploi API
    """

    # We travel through different levels on the ontology
    for lvl1, tuple1 in romev4_handler.ontology.items():

        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(tuple1[0])

        for lvl2, tuple2 in tuple1[1].items():

            print("\n---------------------------------")
            print(tuple2[0])

            for lvl3, tuple3 in tuple2[1].items():

                print("\n.............................")
                print(tuple3[0], "\n")
                
                code_ROME = lvl1 + lvl2 + lvl3

                # For each job category (code OGR):
                for (code_OGR, job_name) in tuple3[1]:

                    file_path = \
                        f"{path}/offers_pole_emploi_scrapped/{code_OGR}.jsonl"
                    
                    # Restart from where we stopped in case of aborted connection
                    # for instance
                    if not os.path.exists(file_path):
                       
                        # We scrap historical offers
                        res = scrapper_pole_emploi.scrap_jobs(
                            appellation_code=code_OGR
                        )
                        
                        nb_job_offers = len(res["resultats"])
                        
                        print(code_OGR, "-", job_name, "-", nb_job_offers)

                        # And save their descriptions in a json file
                        with open(file_path, "w") as f:
                                
                            for r in res["resultats"]:

                                to_save = {
                                    key: r.get(key, "")
                                    for key in (
                                        "id", "intitule", "description"
                                    )
                                }
                                to_save["rome_code"] = code_ROME
                                to_save["appellation_code"] = code_OGR

                                f.write(
                                    json.dumps(to_save) + "\n"
                                )

def sumup_scrapping_pole_emploi(encoder, path):
    """
    Summmarize Pole Emploi job offers scrapping in a single file
    & Precomputes embeddings
    """

    res = {}

    entries = [
        entry for entry
        in os.scandir(f"{path}/offers_pole_emploi_scrapped")
    ]
    for entry in tqdm(entries, total=len(entries)):
        
        with open(entry.path, "r") as f:
            for line in f:
                data = json.loads(line)
                
                if res.get(data["id"], None) is None:
                    res[data["id"]] = {
                        "intitule": data["intitule"] \
                            .replace("\n", " ") \
                            .replace("\xa0", " ") \
                            .replace("&#39;", "'"),
                        "description": data["description"] \
                            .replace("\n", " ") \
                            .replace("\xa0", " ") \
                            .replace("&#39;", "'"),
                        "rome_code": [],
                        "appellation_code": []
                    }

                    embedding_intitule = \
                        encoder.encode(sentences=res[data["id"]]["intitule"])
                    embedding_description = \
                        encoder.encode(sentences=res[data["id"]]["description"])
                    
                    embedding_intitule_base64 = \
                        str(base64.b64encode(embedding_intitule), "utf-8")
                    embedding_description_base64 = \
                        str(base64.b64encode(embedding_description), "utf-8")

                    res[data["id"]]["embedding_intitule_base64"] = \
                        embedding_intitule_base64
                    res[data["id"]]["embedding_description_base64"] = \
                        embedding_description_base64

                if data["rome_code"] not in res[data["id"]]["rome_code"]:
                    res[data["id"]]["rome_code"].append(data["rome_code"])
                if data["appellation_code"] not in res[data["id"]]["appellation_code"]:
                    res[data["id"]]["appellation_code"].append(data["appellation_code"])
    
    with open(f"{path}/offres_pole_emploi_bis.json", "w") as f:
            json.dump(res, f)
