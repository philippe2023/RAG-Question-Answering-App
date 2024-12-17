import json
import os

def load_providers():
    data_path = os.path.join("service_providers.json")
    with open(data_path, "r", encoding="utf-8") as f:
        providers = json.load(f)
    return providers

def find_providers(keywords, location=None):
    providers = load_providers()
    # Basic keyword matching: if provider's keywords intersect with query keywords
    # Convert everything to lowercase for matching
    query_keywords = set(k.lower() for k in keywords)
    matched = []
    for p in providers:
        provider_keywords = set(k.lower() for k in p["keywords"])
        # If intersection is not empty, it means at least one keyword matches
        if query_keywords & provider_keywords:
            if location:
                # If location specified, filter by location
                if p["location"].lower() == location.lower():
                    matched.append(p)
            else:
                # No location filter, accept the match
                matched.append(p)

    return matched