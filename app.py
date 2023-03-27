import random
import json

import torch

import streamlit as st

from model import (
    Encoder, CustomClassifier,
    CLASSES_LVL_1, CLASSES_LVL_1_TO_ID,
    CLASSES_LVL_2, CLASSES_LVL_2_TO_ID,
    CLASSES_LVL_3, CLASSES_LVL_3_TO_ID
)

CLASSES = {
    "LVL_1": CLASSES_LVL_1,
    "LVL_2": CLASSES_LVL_2,
    "LVL_3": CLASSES_LVL_3
}
CLASSES_TO_ID = {
    "LVL_1": CLASSES_LVL_1_TO_ID,
    "LVL_2": CLASSES_LVL_2_TO_ID,
    "LVL_3": CLASSES_LVL_3_TO_ID
}

STATE_DICT_DIR = "model/classifier_statedicts/"

DEVICE = torch.device("cpu")

@st.cache_data()
def load_data():
    data = []
    with open("data.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

@st.cache_data()
def load_models():
    encoder = Encoder(DEVICE)
    classifiers = {}

    for lvl in (1, 2, 3):
        name_lvl = f"LVL_{lvl}"
        
        classifiers[name_lvl] = CustomClassifier(
            DEVICE,
            classes=CLASSES[name_lvl],
            precomputed_embeddings=True
        )
        classifiers[name_lvl].load_from_local_state_dict(f"{STATE_DICT_DIR}{name_lvl}.statedict")
        classifiers[name_lvl].eval()

    return encoder, classifiers

def inference(job_titles, job_descriptions):
    
    with torch.no_grad():
        emb_titles = ENCODER.encode(sentences=job_titles)
        emb_descriptions = ENCODER.encode(sentences=job_descriptions)

        preds = {}
        for lvl, classifier in CLASSIFIERS.items():
            
            preds[lvl] = classifier(
                torch.Tensor(emb_titles),
                torch.Tensor(emb_descriptions)
            )
    
    return preds

def get_res(preds):

    res = {}
    for lvl, predictions in preds.items():

        res[lvl] = [[] for _ in range(len(predictions))]

        sigmoid, softmax = torch.sigmoid(predictions), torch.softmax(predictions, dim=1)

        for i, (sig, soft) in enumerate(zip(sigmoid, softmax)):

            topk_sig = torch.topk(sig, k=3)
            topk_soft = torch.topk(soft, k=3)
            
            for j, (s1, s2) in enumerate(zip(topk_sig.values, topk_soft.values)):
                res[lvl][i].append(
                    (s1.item(), s2.item(), CLASSES[lvl][topk_sig.indices[j].item()])
                )

    return res

def show_random():

    _, _, c, _, _ = st.columns(5)
    with c:
        st.button(label="Shuffle 🔀")

    random_job_offers = random.sample(DATA, 10)

    job_titles = [job_offer["title"] for job_offer in random_job_offers]
    job_descriptions = [job_offer["description"] for job_offer in random_job_offers]

    preds = inference(job_titles, job_descriptions)
    res = get_res(preds)

    for i in range(10):

        st.markdown("---")

        st.write(f"- TITRE: {random_job_offers[i]['title']}")
        st.write(f"- DESCRIPTION: {random_job_offers[i]['description']}")
        
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("#### LEVEL 1")
            for k in range(3):
                st.write(
                    f"""
                    {k+1}: {" - ".join(res['LVL_1'][i][k][2])} \n
                    ➡️ Proba: {round(100 * res['LVL_1'][i][k][0], 1)}% \n
                    ➡️ Proba normalisée: {round(100 * res['LVL_1'][i][k][1], 1)}%
                    """
                )
        
        with c2:
            st.write("#### LEVEL 2")
            for k in range(3):
                st.write(
                    f"""
                    {k+1}: {" - ".join(res['LVL_2'][i][k][2])} \n
                    ➡️ Proba: {round(100 * res['LVL_2'][i][k][0], 1)}% \n
                    ➡️ Proba normalisée: {round(100 * res['LVL_2'][i][k][1], 1)}%
                    """
                )
        
        with c3:
            st.write("#### LEVEL 3")
            for k in range(3):
                st.write(
                    f"""
                    {k+1}: {" - ".join(res['LVL_3'][i][k][2])} \n
                    ➡️ Proba: {round(100 * res['LVL_3'][i][k][0], 1)}% \n
                    ➡️ Proba normalisée: {round(100 * res['LVL_3'][i][k][1], 1)}%
                    """
                )

def show_try():

    job_title = st.text_input(
        "Job Title", "Front-End Developer"
    )
    job_description = st.text_input(
        "Job Description", "**About the role** Side is seeking an experienced React developer to join our team and push our web applications to the next level **Your mission?** Using your React, Typescript and HTML/CSS experiences you’ll work on various topics such as internal front librairies to bring consistency and reusability, internationalization to adapt our product to potential expansion and more. You’ll also use your integration skills to bring responsiveness and lightness throughout the different products we have, from consequent React/Typescript web apps to server-side rendered HTML/CSS landing pages. **What we need your help with** * Own and improve our existing front-end guidelines and libraries (along with our designers) * Build the internationalization & localization components and guidelines * Having a testing mindset: unit tests, code reviews, functional tests, etc. * Improve and unify CI / CD processes accross all front ends * Evangelise front-end integration basics as an habit * Ensure cross browser compatibility & testing as well as responsiveness"
    )

    if job_title and job_description:

        preds = inference([job_title], [job_description])
        res = get_res(preds)

        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("#### LEVEL 1")
            for k in range(3):
                st.write(
                    f"""
                    {k+1}: {" - ".join(res['LVL_1'][0][k][2])} \n
                    ➡️ Proba: {round(100 * res['LVL_1'][0][k][0], 1)}% \n
                    ➡️ Proba normalisée: {round(100 * res['LVL_1'][0][k][1], 1)}%
                    """
                )
        
        with c2:
            st.write("#### LEVEL 2")
            for k in range(3):
                st.write(
                    f"""
                    {k+1}: {" - ".join(res['LVL_2'][0][k][2])} \n
                    ➡️ Proba: {round(100 * res['LVL_2'][0][k][0], 1)}% \n
                    ➡️ Proba normalisée: {round(100 * res['LVL_2'][0][k][1], 1)}%
                    """
                )
        
        with c3:
            st.write("#### LEVEL 3")
            for k in range(3):
                st.write(
                    f"""
                    {k+1}: {" - ".join(res['LVL_3'][0][k][2])} \n
                    ➡️ Proba: {round(100 * res['LVL_3'][0][k][0], 1)}% \n
                    ➡️ Proba normalisée: {round(100 * res['LVL_3'][0][k][1], 1)}%
                    """
                )
    
if __name__ == "__main__":

    st.set_page_config(
        page_title="Noese - App Demo Classifier JobType",
        page_icon="📖",
        layout="wide"
    )

    DATA = load_data()
    ENCODER, CLASSIFIERS = load_models()

    #%% Main Page
    st.title("JobType Classifier")
    st.write(
        """
        Observe how the classifier performs on different levels of RomeV4's ontology.
        Classifier uses both job title and job description.
        """)

    #%% Side bar
    st.sidebar.title("Noese - JobType Classifier")
    st.sidebar.write(
        """
        Small & simple app to check how perform JobType classifier on different
        ontology levels.
        \n(Level 1, 2 & 3 for now)
        """
    )   

    page_names_to_funcs = {
        "Random Job Offers": show_random,
        "Try with my Job Offer": show_try,
    }
    selected_page = st.sidebar.selectbox("Select mode", page_names_to_funcs.keys())

    page_names_to_funcs[selected_page]()

    

