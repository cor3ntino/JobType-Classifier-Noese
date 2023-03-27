import pandas as pd

def load_xlsx_df(path: str):
    
    return pd.read_excel(
        path,
        sheet_name="Arbo Principale 14-06-2021"
    )
    
class RomeV4Handler:

    def __init__(self, data_path):

        self.df = load_xlsx_df(data_path + "ROMEV4.xlsx")
        self.build_ontology()

    def build_ontology(self):

        self.ontology = {}

        for _, row in self.df.iterrows():

            if len(row["Code OGR"]) > 1:
                self.ontology[
                    row["Niveau 1"]
                ][1][
                    row["Niveau 2"]
                ][1][
                    row["Niveau 3"]
                ][1].append(
                    (row["Code OGR"], row["Titre"])
                )
            
            elif len(row["Niveau 3"]) > 1:
                self.ontology[
                    row["Niveau 1"]
                ][1][
                    row["Niveau 2"]
                ][1][
                    row["Niveau 3"]
                ] = (row["Titre"], [])

            elif len(row["Niveau 2"]) > 1:
                self.ontology[
                    row["Niveau 1"]
                ][1][
                    row["Niveau 2"]
                ] = (row["Titre"], {})

            else:
                self.ontology[
                    row["Niveau 1"]
                ] = (row["Titre"], {})

    def get_jobs_title(self):

        jobs = []

        for i, row in self.df.iterrows():
            if row["Code OGR"] != " ":
                jobs.append((row["Code OGR"], row["Titre"]))
        
        return jobs

    def get_jobs_category(self):

        categories = []

        for lvl1, tuple1 in self.ontology.items():

            for lvl2, tuple2 in tuple1[1].items():

                for lvl3, tuple3 in tuple2[1].items():

                    categories.append((lvl1+lvl2+lvl3, tuple3[0]))
        
        return categories
                    

