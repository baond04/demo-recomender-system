"""
src/rag_conversational/__init__.py
Module: GNN + GCL + RAG Personalized Conversational Recommender
================================================================
Tích hợp ba thành phần:
  1. GNN+GCL (LightGCN + SimGCL): Khai thác tín hiệu cộng tác dài hạn
  2. RAG (Retrieval-Augmented Generation): Truy xuất ngữ cảnh hội thoại
  3. LLM (Gemini API): Sinh phản hồi tự nhiên, cá nhân hóa

Pipeline 3-Phase:
  Phase 1: Song song — RAG retrieval + GNN+GCL recommendation
  Phase 2: Fusion  — Kết hợp ngữ cảnh + items vào prompt
  Phase 3: Generate — LLM sinh phản hồi + cập nhật CSDL
"""

from .conversation_store import ConversationStore
from .rag_retriever import RAGRetriever
from .gnn_gcl_adapter import GNNGCLAdapter
from .prompt_builder import PromptBuilder
from .response_generator import ResponseGenerator
from .conversational_recommender import ConversationalRecommender

__all__ = [
    'ConversationStore',
    'RAGRetriever',
    'GNNGCLAdapter',
    'PromptBuilder',
    'ResponseGenerator',
    'ConversationalRecommender',
]

__version__ = '1.0.0'
