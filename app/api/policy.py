from app.core.rag_system import RAGSystem
from app.core.dependencies import get_rag_system
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException


router = APIRouter()
