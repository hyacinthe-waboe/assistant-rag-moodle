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
import json
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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


def _etat_path(course_id: str) -> str:
    dossier = os.path.join(config.DATA_DIR, rag._valider_course_id(course_id))
    os.makedirs(dossier, exist_ok=True)
    return os.path.join(dossier, "index_status.json")


def _sauver_etat_course(course_id: str, etat: dict) -> None:
    chemin = _etat_path(course_id)
    dossier = os.path.dirname(chemin)
    fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".status.tmp")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(etat, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _charger_etat_course(course_id: str) -> dict:
    chemin = _etat_path(course_id)
    if not os.path.exists(chemin):
        return {}
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _etat_public(job: dict, include_finished: bool = False) -> dict:
    result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
    public = {
        "status": str(job.get("status", "idle")),
        "stage": str(job.get("stage", "idle")),
        "progress": int(job.get("progress", 0)),
        "message": str(job.get("message", "")),
        "error": str(job.get("error", "")),
        "started_at": str(job.get("started_at", "")),
        "updated_at": str(job.get("updated_at", "")),
        "fichiers": int(job.get("files", result.get("fichiers", 0)) or 0),
        "fichiersexploitables": int(result.get("fichiers_exploitables", 0)),
        "fichiers_ocr": int(result.get("fichiers_ocr", 0)),
        "pages_ocr": int(result.get("pages_ocr", 0)),
        "chunks": int(result.get("chunks", 0)),
        "tokens": int(result.get("tokens_embedding", 0)),
    }

    if include_finished:
        public["result"] = result
        return public

    if public["status"] in {"completed", "failed"}:
        public["last_status"] = public["status"]
        public["last_stage"] = public["stage"]
        public["last_progress"] = public["progress"]
        public["last_message"] = public["message"]
        public["last_error"] = public["error"]
        public["last_started_at"] = public["started_at"]
        public["last_updated_at"] = public["updated_at"]
        public["last_result"] = result
        public["status"] = "idle"
        public["stage"] = "idle"
        public["message"] = "Aucune indexation en cours."
        public["error"] = ""
        public["started_at"] = ""
        public["updated_at"] = ""
        public["result"] = {}
    return public


def _verifier_auth(request: Request) -> None:
    """Autorise uniquement les requêtes qui présentent le token partagé."""
    token_attendu = config.BACKEND_SHARED_TOKEN.strip()
    if not token_attendu:
        return

    token_recu = request.headers.get("x-rag-token", "").strip()
    if not token_recu:
        auth = request.headers.get("authorization", "")
        prefix = "bearer "
        if auth.lower().startswith(prefix):
            token_recu = auth[len(prefix):].strip()

    if token_recu != token_attendu:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification RAG invalide ou manquante.",
        )


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
        _sauver_etat_course(course_id, _etat_public(dict(index_jobs.get(course_id, {})), include_finished=True))
    except Exception as erreur:
        _modifier_job(
            course_id,
            status="failed",
            stage="failed",
            message="L'indexation a échoué.",
            error=str(erreur),
        )
        _sauver_etat_course(course_id, _etat_public(dict(index_jobs.get(course_id, {})), include_finished=True))
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


# ── Schémas ───────────────────────────────────────────────────────────────────

class MessageHistorique(BaseModel):
    role:    Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class AskRequest(BaseModel):
    course_id: str = Field(pattern=r"^\d+$", max_length=20)
    question:  str = Field(min_length=1, max_length=2000)
    k:         int = Field(default=config.TOP_K, ge=1, le=20)
    history:   list[MessageHistorique] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def nettoyer_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide.")
        return question


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
def health(request: Request):
    _verifier_auth(request)
    return {"status": "ok", "provider": config.PROVIDER}


@app.post("/index/{course_id}", status_code=202)
async def index_course(course_id: str, request: Request, background_tasks: BackgroundTasks):
    """Enregistre les PDF puis lance la reconstruction en arrière-plan."""
    _verifier_auth(request)
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
def index_status(course_id: str, request: Request):
    """Renvoie l'état de la dernière indexation demandée pour un cours."""
    _verifier_auth(request)
    include_finished = request.query_params.get("include_finished", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        course_id = rag._valider_course_id(course_id)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur

    with index_jobs_lock:
        job = dict(index_jobs.get(course_id, {}))

    if not job:
        etat_persiste = _charger_etat_course(course_id)
        if etat_persiste:
            return {"course_id": course_id, **_etat_public(etat_persiste, include_finished=include_finished)}
        return {
            "course_id": course_id,
            "status": "idle",
            "stage": "idle",
            "progress": 100 if rag.index_existe(course_id) else 0,
            "message": "Aucune indexation en cours.",
            "result": {},
            "error": "",
            "started_at": "",
            "updated_at": "",
            "last_status": "",
            "last_stage": "",
            "last_progress": 0,
            "last_message": "",
            "last_error": "",
            "last_started_at": "",
            "last_updated_at": "",
            "last_result": {},
        }

    if not include_finished and job.get("status") in {"completed", "failed"}:
        return {"course_id": course_id, **_etat_public(job, include_finished=False)}

    return {"course_id": course_id, **job}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    """Répond à une question en s'ancrant sur les ressources indexées."""
    _verifier_auth(request)
    reponse_locale = rag.repondre_message_courant(req.question)
    if reponse_locale:
        return reponse_locale

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
