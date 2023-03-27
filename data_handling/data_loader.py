import pandas as pd

def load_csv_df(path: str):
    
    return pd.read_csv(path, sep=",")

def generator_csv_df(path: str, chuncksize: int = 10000):
    """
    Useful for large CSV files if computer raises OutOfMemory
    """

    with pd.read_csv(path, sep=",", chunksize=chuncksize) as reader:
        for chunk in reader:
            yield(chunk)

class DataLoader:

    def __init__(self, data_path):

        resource_path = data_path + "resource"

        self.df_job             = load_csv_df(resource_path + "/JobResource.csv")
        self.df_contract_type   = load_csv_df(resource_path + "/ContractTypeResource.csv")
        self.df_company         = load_csv_df(resource_path + "/CompanyResource.csv")
        self.df_offer_exclusive = load_csv_df(resource_path + "/OfferExclusiveResource.csv")
        self.df_offer_status    = load_csv_df(resource_path + "/OfferStatusResource.csv")
        self.df_offer_type      = load_csv_df(resource_path + "/OfferTypeResource.csv")

    def get_job_descriptions(self):

        return [
            " ".join(
                description \
                    .replace("## Descriptif du poste", "") \
                    .split()
                )
            if isinstance(description, str)
            else ""
            for description in self.df_job["description"]
        ]
    
    def get_job_profiles(self):

        return [
            " ".join(
                profil \
                    .replace("## Profil recherché", "") \
                    .split()
                )
            if isinstance(profil, str)
            else ""
            for profil in self.df_job["profil"]
        ]
    
    def get_job_titles(self):

        return [
            " ".join(
                title \
                    .split()
                )
            if isinstance(title, str)
            else ""
            for title in self.df_job["title"]
        ]
