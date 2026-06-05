# mlops-llm-hallucination-type-detection-project

LLM hallucination type classifier (none / fabrication / confusion) served via FastAPI.

## Run locally (no Docker)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
pip install -e .
python run_api.py            # http://127.0.0.1:8000
```

## Run with Docker

The image is multi-stage: it installs a CPU-only PyTorch wheel, then copies
the trained model from `models/saved_distilbert_hallucination_model/` into a
slim runtime layer. The API listens on `0.0.0.0:8000`.

### Build

```bash
docker build -t hallucination-api:latest .
```

### Run (plain Docker)

```bash
docker run --rm -p 8000:8000 --name hallucination-api hallucination-api:latest
```

Then open:

- UI: <http://localhost:8000/>
- Health: <http://localhost:8000/health>
- Docs: <http://localhost:8000/docs>

### Run (docker compose)

```bash
docker compose up --build
```

To mount a different model checkpoint at runtime, uncomment the `volumes:`
block in `docker-compose.yml` and rebuild.

### Useful endpoints

| Method | Path           | Purpose                                    |
| ------ | -------------- | ------------------------------------------ |
| GET    | `/`            | HTML UI for quick testing                  |
| GET    | `/health`      | Liveness + which backend is loaded         |
| POST   | `/predict`     | JSON inference: `{"text": "..."}`          |
| POST   | `/predict/form`| HTML form submission used by the UI        |
| GET    | `/docs`        | Auto-generated Swagger UI                  |

### Image size notes

- The CPU-only torch wheel from `https://download.pytorch.org/whl/cpu` is
  used to avoid shipping the ~1.5 GB of CUDA libraries.
- `.dockerignore` excludes `.venv`, `data/`, `notebooks/`, `docs/`, etc.