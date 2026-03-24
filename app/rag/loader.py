from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List
from uuid import uuid4


CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _read_text(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> List[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_PATTERN.split(text) if sentence.strip()]
    return sentences or [text]


def _chunk_sentences(
    sentences: List[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    chunks: List[str] = []
    sentence_words = [sentence.split() for sentence in sentences]

    start_index = 0
    total_sentences = len(sentence_words)

    while start_index < total_sentences:
        current_chunk_words: List[str] = []
        end_index = start_index

        while end_index < total_sentences:
            next_sentence_words = sentence_words[end_index]
            proposed_size = len(current_chunk_words) + len(next_sentence_words)

            if current_chunk_words and proposed_size > chunk_size:
                break

            current_chunk_words.extend(next_sentence_words)
            end_index += 1

        if not current_chunk_words:
            current_chunk_words = sentence_words[start_index][:chunk_size]
            end_index = start_index + 1

        chunks.append(" ".join(current_chunk_words).strip())

        if end_index >= total_sentences:
            break

        overlap_words: List[str] = []
        rewind_index = end_index - 1
        while rewind_index >= start_index and len(overlap_words) < overlap:
            overlap_words = sentence_words[rewind_index] + overlap_words
            rewind_index -= 1

        overlap_word_count = min(len(overlap_words), overlap)
        accumulated_words = 0
        next_start_index = end_index - 1

        for index in range(end_index - 1, start_index - 1, -1):
            accumulated_words += len(sentence_words[index])
            next_start_index = index
            if accumulated_words >= overlap_word_count:
                break

        start_index = next_start_index if next_start_index > start_index else end_index

    return chunks


def load_and_chunk(file_path: str) -> List[Dict]:
    source_path = Path(file_path)
    normalized_text = _normalize_text(_read_text(file_path))
    sentences = _split_sentences(normalized_text)
    chunk_texts = _chunk_sentences(sentences)

    chunks: List[Dict] = []
    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunks.append(
            {
                "id": str(uuid4()),
                "text": chunk_text,
                "metadata": {
                    "source": source_path.name,
                    "chunk_index": chunk_index,
                },
            }
        )

    sample_preview = chunks[0]["text"][:200] if chunks else ""
    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk preview: {sample_preview}")

    return chunks
