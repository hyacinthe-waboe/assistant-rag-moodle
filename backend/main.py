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
import tempfile

from fastapi import FastAPI, HTTPException, Request
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


@app.post("/index/{course_id}")
async def index_course(course_id: str, request: Request):
    """Construit ou reconstruit l'index vectoriel d'un cours."""
    form    = await request.form()
    uploads = [v for v in form.values() if isinstance(v, StarletteUploadFile)]

    if not uploads:
        raise HTTPException(status_code=400, detail="Aucun fichier reçu.")

    fichiers_pdf = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in uploads:
            if not f.filename.lower().endswith(".pdf"):
                continue
            chemin = os.path.join(tmp, os.path.basename(f.filename))
            with open(chemin, "wb") as out:
                out.write(await f.read())
            fichiers_pdf.append((chemin, os.path.basename(f.filename)))

        if not fichiers_pdf:
            raise HTTPException(status_code=400, detail="Aucun PDF valide reçu.")

        try:
            resume = rag.construire_index(course_id, fichiers_pdf, provider)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return resume


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
