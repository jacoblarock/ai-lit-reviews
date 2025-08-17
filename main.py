import requests
import xmltodict
import ollama
import json
from arxiv import Client, Search

model = "qwen3:14b"

def create_queries(topic: str) -> list[str]:
    with open("prompts/create_queries.txt") as file:
        prompt = file.read().replace("TOPIC", topic)
    while True:
        resp = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        try:
            content = resp["message"]["content"]
            start_index = content.rfind("[")
            end_index = content.rfind("]") + 1
            content = content[start_index:end_index]
            return json.loads(content)
        except:
            print(resp["message"]["content"])
            print("retrying")

def query_articles(search_terms: list[str]):
    client = Client()
    results = []
    for term in search_terms:
        print(term)
        search = Search(
            query=term,
            max_results=10
        )
        resp = list(client.results(search))
        print(len(resp))
        results += resp
    return results

def main():
    topic = "Explainable AI in the area of audio deepfake detection."
    article_metadata = []
    while len(article_metadata) < 20:
        queries = create_queries(topic)
        article_metadata += query_articles(queries)
    # print(json.dumps(article_metadata, indent=2))

if __name__ == "__main__":
    main()