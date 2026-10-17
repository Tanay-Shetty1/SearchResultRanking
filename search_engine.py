
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = [
    {
        "id": 1,
        "title": "Introduction to Computer Networks",
        "content": "Computer networks connect devices and allow data communication using protocols, routers and switches."
    },
    {
        "id": 2,
        "title": "Network Security Fundamentals",
        "content": "Network security protects computer networks from cyberattacks using firewalls, encryption and access control."
    },
    {
        "id": 3,
        "title": "Information Retrieval Basics",
        "content": "Information retrieval searches a collection of documents and ranks results according to their relevance to a query."
    },
    {
        "id": 4,
        "title": "Search Engine Ranking",
        "content": "Search engines rank documents using TF-IDF, cosine similarity and relevance scores to find useful information."
    },
    {
        "id": 5,
        "title": "Cybersecurity and Encryption",
        "content": "Cybersecurity uses encryption, authentication and security controls to protect systems, networks and information."
    },
    {
        "id": 6,
        "title": "TCP and UDP Protocols",
        "content": "TCP provides reliable data delivery while UDP supports faster connectionless communication between applications."
    },
    {
        "id": 7,
        "title": "Introduction to Science",
        "content": "Science is the systematic study of the natural world through observation, experiments, evidence and scientific methods. It includes physics, chemistry, biology and environmental science."
    },
    {
        "id": 8,
        "title": "Introduction to Biology",
        "content": "Biology is the scientific study of living organisms, including plants, animals, cells, genetics and ecosystems."
    },
    {
        "id": 9,
        "title": "Introduction to Physics",
        "content": "Physics studies matter, energy, motion, forces, electricity and the fundamental laws of the universe."
    },
    {
        "id": 10,
        "title": "Introduction to Chemistry",
        "content": "Chemistry studies elements, compounds, atoms, molecules and chemical reactions."
    },
    {
        "id": 11,
        "title": "Introduction to Computer Science",
        "content": "Computer science is the study of computers, algorithms, programming, software development, data structures, databases and computational systems."
    },
]


def search_documents(query, top_k=5):
    query = query.strip()

    if not query:
        return []

    # Create separate TF-IDF indexes for titles and content.
    titles = [doc["title"] for doc in DOCUMENTS]
    contents = [doc["content"] for doc in DOCUMENTS]

    title_vectorizer = TfidfVectorizer(stop_words="english")
    content_vectorizer = TfidfVectorizer(stop_words="english")

    title_vectors = title_vectorizer.fit_transform(titles)
    content_vectors = content_vectorizer.fit_transform(contents)

    title_query = title_vectorizer.transform([query])
    content_query = content_vectorizer.transform([query])

    title_scores = cosine_similarity(
        title_query, title_vectors
    ).flatten()

    content_scores = cosine_similarity(
        content_query, content_vectors
    ).flatten()

    # Extract meaningful query terms using the same analyzer.
    analyzer = title_vectorizer.build_analyzer()
    query_terms = set(analyzer(query))

    results = []

    for index, doc in enumerate(DOCUMENTS):
        title_terms = set(analyzer(doc["title"]))
        content_terms = set(analyzer(doc["content"]))

        matched_terms = query_terms & (title_terms | content_terms)

        coverage = (
            len(matched_terms) / len(query_terms)
            if query_terms else 0.0
        )

        # Weighted relevance score.
        score = (
            0.60 * title_scores[index]
            + 0.25 * content_scores[index]
            + 0.15 * coverage
        )

        if score > 0:
            results.append({
                **doc,
                "score": round(float(score) * 100, 2),
                "coverage": round(coverage * 100, 2)
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:max(0, top_k)]


if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = search_documents(query)

    if not results:
        print("\nNo matching documents found.")
    else:
        print("\nRanked Search Results")
        print("=" * 40)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Title: {result['title']}")
            print(f"Relevance score: {result['score']}%")
            print(f"Query coverage: {result['coverage']}%")
            print(f"Content: {result['content']}")
