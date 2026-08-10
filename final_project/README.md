# Chatbot deployment

## Files
- `main.py` — Streamlit app (login/register/logout, chat switching, chat UI)
- `auth.py` — password hashing + register/login
- `db.py` — Cloud SQL (Postgres) access: users, conversations, messages
- `chat.py` — LangChain + Chroma retrieval chain
- `ingest.py` — rebuild the Chroma store from a folder of source docs
- `Dockerfile`, `requirements.txt`

## Environment variables

| Variable | Purpose | Example |
|---|---|---|
| `DB_USER` | Postgres user | `postgres` |
| `DB_PASSWORD` | Postgres password | (from Secret Manager) |
| `DB_NAME` | Database name | `users_and_conversations` |
| `DB_HOST` | Local dev only | `localhost` |
| `DB_PORT` | Local dev only | `5432` |
| `INSTANCE_CONNECTION_NAME` | Cloud Run only, enables Unix socket connection | `project:region:instance` |
| `CHROMA_PERSIST_DIR` | Where Chroma reads/writes | `/mnt/chroma/chroma_db` on Cloud Run, `./chroma_db` locally |
| `API_KEY` | LLM + embeddings provider | (from Secret Manager) |

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export DB_USER=postgres DB_PASSWORD=yourpassword DB_NAME=users_and_conversations
export DB_HOST=localhost DB_PORT=5432
export CHROMA_PERSIST_DIR=./chroma_db
export API_KEY=sk-...

streamlit run main.py
```

## Deploy to Cloud Run

See the deployment steps discussed in chat — in short:
1. Push `./chroma_db` to a GCS bucket, mount it into Cloud Run with `--add-volume`.
2. Build & push the Docker image to Artifact Registry.
3. Deploy with `--add-cloudsql-instances`, `--set-secrets`, `--session-affinity`, `--max-instances 1`.

## Re-ingesting docs

```bash
python ingest.py --source ./new_docs
gsutil -m rsync -r ./chroma_db gs://YOUR_PROJECT-chroma-db/chroma_db
```

## Test the full stack locally with Docker Compose

This spins up the app + a local Postgres so you can test everything
before touching GCP.

```bash
cp ..env .env   # fill in API_KEY at minimum
docker compose up --build
```

Visit `http://localhost:8501`. Data persists in a Docker volume
(`pgdata`) across restarts; run `docker compose down -v` to wipe it.

## Automated build + deploy with Cloud Build

`cloudbuild.yaml` builds the image, pushes it to Artifact Registry, and
deploys to Cloud Run in one shot.

One-time setup — give Cloud Build's service account permission to deploy
to Cloud Run and read secrets:

```bash
PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

Then trigger a build + deploy:

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_INSTANCE_CONNECTION_NAME="PROJECT_ID:southamerica-west1:users-and-conversations",_CHROMA_BUCKET="chroma_db_users_and_conversations"
```

To build automatically on every `git push`, connect this repo to Cloud
Build via a GitHub trigger (Cloud Console → Cloud Build → Triggers →
Connect Repository) pointing at `cloudbuild.yaml`.
