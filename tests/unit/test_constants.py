"""Unit tests for constant tables and lookup helpers (Task 3.2).

Verifies skill synonym mapping, degree/tier/field mappings, and weight sums.

_Requirements: 3.4_
"""

import pytest

from ranking.constants import (
    WEIGHTS,
    degree_to_level,
    field_relevance,
    skill_to_group,
    tier_value,
)


@pytest.mark.parametrize(
    "variant, expected_group",
    [
        ("Pinecone", "vector_databases"),
        ("Milvus", "vector_databases"),
        ("Weaviate", "vector_databases"),
        ("ChromaDB", "vector_databases"),
        ("FAISS", "vector_databases"),
        ("Qdrant", "vector_databases"),
        ("LoRA", "llm_finetuning"),
        ("QLoRA", "llm_finetuning"),
        ("PEFT", "llm_finetuning"),
        ("python", "python"),
        ("pytorch", "python"),
        ("ndcg", "evaluation_frameworks"),
        ("mrr", "evaluation_frameworks"),
        ("map", "evaluation_frameworks"),
        ("kafka", "distributed_systems"),
        ("spark", "distributed_systems"),
    ],
)
def test_skill_to_group_maps_known_variants(variant, expected_group):
    assert skill_to_group(variant) == expected_group


@pytest.mark.parametrize(
    "variant, expected_group",
    [
        ("PINECONE", "vector_databases"),
        ("lora", "llm_finetuning"),
        ("Python", "python"),
        ("NDCG", "evaluation_frameworks"),
        ("Kafka", "distributed_systems"),
    ],
)
def test_skill_to_group_is_case_insensitive(variant, expected_group):
    assert skill_to_group(variant) == expected_group


def test_skill_to_group_unknown_returns_none():
    assert skill_to_group("Underwater Basket Weaving") is None


def test_skill_to_group_empty_returns_none():
    assert skill_to_group("") is None


@pytest.mark.parametrize(
    "degree, expected",
    [
        ("Ph.D", 1.0),
        ("PhD", 1.0),
        ("M.Tech", 0.8),
        ("M.Sc", 0.8),
        ("B.Tech", 0.6),
        ("B.E.", 0.6),
    ],
)
def test_degree_to_level(degree, expected):
    assert degree_to_level(degree) == expected


@pytest.mark.parametrize(
    "tier, expected",
    [
        ("tier_1", 1.0),
        ("unknown", 0.25),
    ],
)
def test_tier_value(tier, expected):
    assert tier_value(tier) == expected


@pytest.mark.parametrize(
    "field, expected",
    [
        ("Computer Science", 1.0),
        ("Mathematics", 0.5),
        ("Underwater Basket Weaving", 0.2),
    ],
)
def test_field_relevance(field, expected):
    assert field_relevance(field) == expected


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
