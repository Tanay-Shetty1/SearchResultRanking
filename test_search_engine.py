
from search_engine import search_documents, DOCUMENTS


def test_empty_query():
    assert search_documents("") == []
    assert search_documents("   ") == []


def test_computer_science_ranking():
    results = search_documents("computer science")
    assert results
    assert results[0]["title"] == "Introduction to Computer Science"


def test_computer_networks_ranking():
    results = search_documents("computer networks")
    assert results
    assert results[0]["title"] == "Introduction to Computer Networks"


def test_network_security_ranking():
    results = search_documents("network security")
    assert results
    assert results[0]["title"] == "Network Security Fundamentals"


def test_results_sorted_by_score():
    results = search_documents("science")
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limit():
    results = search_documents("computer", top_k=2)
    assert len(results) <= 2


def test_unknown_query():
    assert search_documents("qzxvplm") == []


def test_document_count():
    assert len(DOCUMENTS) == 11
