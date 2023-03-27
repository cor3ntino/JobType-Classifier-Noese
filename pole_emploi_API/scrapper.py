import time
import json

import requests as rq

class Scrapper:

    _ID = "PAR_test_2c3b47505ef0c09542e27e2658813bad0d1196e1790ce5734df446d566730574"
    _PASS = "33ad8fef503f29967082b0952999c435aa91d88b36acfca97a3c30eaf05d46c5"

    def __init__(self):

        self.url_access_token = \
            "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token"
        self.url_job_search = \
            "https://api.pole-emploi.io/partenaire/offresdemploi/v2/offres/search/"

        self.get_access_token()
        self.last_hit_API_time = 0.
        
    def get_access_token(self):
        """
        Get Access Token from Pole Emploi.
        """

        resp = rq.post(
            self.url_access_token,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Cache-Control": "no-store",
                "Pragma": "no-cache"
            },
            params={
                "realm": "/partenaire"
            },
            data={
                "grant_type": "client_credentials",
                "client_id": self._ID,
                "client_secret": self._PASS,
                "scope": "api_offresdemploiv2 o2dsoffre",
            }
        )

        self.access_token_time = time.time()
        self.access_token = resp.json()

    def renew_access_token(self):
        """
        If access token expired, get new token
        """

        new_time = time.time()
        self.access_token['expires_in'] -= new_time - self.access_token_time
        self.access_token_time = new_time

        # If less than 20 seconds remaining, get new access token
        if self.access_token['expires_in'] < 20.:
            self.get_access_token()

    def scrap_jobs(self, rome_code: str = None, appellation_code: str = None):
        """
        Request Pole Emploi's API to retrieve job offer with specific ROME code
        or appellation.
        """

        self.renew_access_token()
        self.control_hit_API()

        # Pass arguments to GET
        params = {}
        if rome_code:
            params["codeROME"] = rome_code
        if appellation_code:
            params["appellation"] = appellation_code

        resp = rq.get(
            self.url_job_search,
            headers={
                "Authorization": f"Bearer {self.access_token['access_token']}"
            },
            params=params
        )

        self.last_hit_API_time = time.time()

        try:
            return resp.json()
        
        except json.decoder.JSONDecodeError:
            print("No valid data")
            return {
                "resultats": []
            }
    
    def control_hit_API(self):
        """
        To request API at most twice per second
        """

        min_interval = 0.5 # 0.5 seconds

        wait_time = time.time() - self.last_hit_API_time

        if wait_time < min_interval:
            time.sleep(min_interval - wait_time)
