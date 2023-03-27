# Noese - Classifier JobType

Classifying job offers from different websites (e.g. WelcomeToTheJungle, ..) into categories available in RomeV4; such categories are organized into an ontology with 4-levels of depth. 

## Use locally

Create a virtual environment and install dependencies:

```bash
virtualenv -p python3.10 env
ln -s env/bin/activate
source activate

pip install -r requirements.txt
```

Run demo through Streamlit app:
```bash
streamlit run app.py --server.port 8501
```
