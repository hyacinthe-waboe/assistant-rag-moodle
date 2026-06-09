# -*- coding: utf-8 -*-
"""
API FastAPI — point d'entrée HTTP du backend RAG Moodle.

Endpoints :
  GET  /health            → état du service
  POST /index/{course_id} → construit l'index d'un cours à partir de PDF
  POST /ask               → répond à une question ancrée sur les ressources

Lancement :
    uvicorn main:app --host 127.0.0.1 --port 8000
"""

import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

import config
import rag
from providers import obtenir_provider

app = FastAPI(title="RAG Moodle Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

provider = obtenir_provider()
index_jobs = {}
index_jobs_lock = threading.Lock()


def _maintenant() -> str:
    return datetime.now(timezone.utc).isoformat()


def _modifier_job(course_id: str, **valeurs) -> None:
    with index_jobs_lock:
        job = index_jobs.setdefault(course_id, {})
        job.update(valeurs)
        job["updated_at"] = _maintenant()


def _executer_indexation(course_id: str, dossier: str,
                        fichiers_pdf: list[tuple[str, str]]) -> None:
    def progression(etape: str, pourcentage: int, message: str) -> None:
        _modifier_job(
            course_id,
            status="running",
            stage=etape,
            progress=pourcentage,
            message=message,
        )

    try:
        _modifier_job(
            course_id,
            status="running",
            stage="extraction",
            progress=1,
            message="Démarrage de l'indexation.",
        )
        resume = rag.construire_index(
            course_id,
            fichiers_pdf,
            provider,
            progression,
        )
        _modifier_job(
            course_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Indexation terminée.",
            result=resume,
            error="",
        )
    except Exception as erreur:
        _modifier_job(
            course_id,
            status="failed",
            stage="failed",
            message="L'indexation a échoué.",
            error=str(erreur),
        )
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


# ── Schémas ───────────────────────────────────────────────────────────────────

class MessageHistorique(BaseModel):
    role:    str   # "user" ou "assistant"
    content: str


class AskRequest(BaseModel):
    course_id: str
    question:  str
    k:         int            = config.TOP_K
    history:   list[MessageHistorique] = Field(default_factory=list)


class PassageItem(BaseModel):
    source: str
    page:   int | str
    texte:  str


class AskResponse(BaseModel):
    reponse:  str
    sources:  list[str]
    passages: list[PassageItem] = Field(default_factory=list)
    tokens:   int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "provider": config.PROVIDER}


@app.post("/index/{course_id}", status_code=202)
async def index_course(course_id: str, request: Request, background_tasks: BackgroundTasks):
    """Enregistre les PDF puis lance la reconstruction en arrière-plan."""
    try:
        course_id = rag._valider_course_id(course_id)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur

    with index_jobs_lock:
        job = index_jobs.get(course_id)
        if job and job.get("status") in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail="Une indexation est déjà en cours pour ce cours.",
            )

    form    = await request.form()
    uploads = [v for v in form.values() if isinstance(v, StarletteUploadFile)]

    if not uploads:
        raise HTTPException(status_code=400, detail="Aucun fichier reçu.")

    fichiers_pdf = []
    tmp = tempfile.mkdtemp(prefix=f"ragchat_{course_id}_")
    try:
        for f in uploads:
            if not f.filename.lower().endswith(".pdf"):
                continue
            chemin = os.path.join(tmp, os.path.basename(f.filename))
            with open(chemin, "wb") as out:
                out.write(await f.read())
            fichiers_pdf.append((chemin, os.path.basename(f.filename)))

        if not fichiers_pdf:
            raise HTTPException(status_code=400, detail="Aucun PDF valide reçu.")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    _modifier_job(
        course_id,
        status="queued",
        stage="queued",
        progress=0,
        message=f"{len(fichiers_pdf)} PDF reçus, indexation en attente.",
        files=len(fichiers_pdf),
        result={},
        error="",
        started_at=_maintenant(),
    )
    background_tasks.add_task(_executer_indexation, course_id, tmp, fichiers_pdf)
    return {
        "course_id": course_id,
        "status": "queued",
        "progress": 0,
        "fichiers": len(fichiers_pdf),
        "message": f"{len(fichiers_pdf)} PDF reçus.",
    }


@app.get("/index/{course_id}/status")
def index_status(course_id: str):
    """Renvoie l'état de la dernière indexation demandée pour un cours."""
    try:
        course_id = rag._valider_course_id(course_id)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur

    with index_jobs_lock:
        job = dict(index_jobs.get(course_id, {}))

    if not job:
        return {
            "course_id": course_id,
            "status": "idle",
            "stage": "idle",
            "progress": 100 if rag.index_existe(course_id) else 0,
            "message": "Aucune indexation en cours.",
            "result": {},
            "error": "",
        }
    return {"course_id": course_id, **job}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Répond à une question en s'ancrant sur les ressources indexées."""
    try:
        index_present = rag.index_existe(req.course_id)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur

    if not index_present:
        raise HTTPException(
            status_code=404,
            detail=f"Cours {req.course_id} non indexé.",
        )
    # Convertit les objets Pydantic en dicts simples pour rag.py
    history = [{"role": m.role, "content": m.content} for m in req.history]
    return rag.repondre(req.course_id, req.question, req.k, provider, history)
