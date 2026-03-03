# Purwadhika Final Project Submission
This repo is intended to be used for the final project submission of Purwadhika.
It is owned by Christopher, Iqbal, and Wisnunandar.

## Installation
```shell
pip3 install -r api/requirements.txt
pip3 install -r web/requirements.txt
```

## Running API
- First, you need to authenticate with Google Cloud cli. You 
  can follow the instructions [here](https://docs.cloud.google.com/sdk/docs/install-sdk#windows).
- Run `gcloud init` to initialize the CLI.
- Run `gcloud auth application-default login` to authenticate.
- Create `.env` file in the `api` directory with the following contents:
```dotenv
API_KEY=example-api-key
GCS_BUCKET=jcaieh-finpro
```
- Finally, run `uvicorn main:app --host 0.0.0.0 --port 8154` in the `api` directory.

## Running Web
- Create `.streamlit/secrets.toml` file with the following contents in `web` directory:
```toml
DISCORD_CHANNEL_NAME = "our-discord-channel-name" # this will be used in the first page of the streamlit app
REST_API_BASE_URL = "http://127.0.0.1:8154"
REST_API_KEY = "example-api-key"
```
- Then, run `streamlit run main.py` in the `web` directory.
