
from flask import Flask, render_template, request
from search_engine import search_documents, DOCUMENTS
from time import perf_counter

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    results = []
    search_time = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        start = perf_counter()
        results = search_documents(query)
        search_time = round((perf_counter() - start) * 1000, 2)

    return render_template(
        "index.html",
        query=query,
        results=results,
        search_time=search_time,
        total_documents=len(DOCUMENTS)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)