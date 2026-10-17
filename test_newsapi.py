import requests

API_KEY = "4fcb9d18b0ba438a8cce82128b3aec60"

def search_news(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Test karte hain
result = search_news("todays news")
#result = search_news("artificial intelligence")
#result = search_news("cybersecurity")
#result = search_news("india")
for article in result.get("articles", []):
    print(article["title"], "-", article["source"]["name"])