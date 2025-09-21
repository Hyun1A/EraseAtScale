import csv, re, time
import requests
from pathlib import Path

INPUT_CSV = "positives.csv"   # 업로드한 1000명 CSV
OUT_DIR = "classified"

WIKI_API = "https://en.wikipedia.org/w/api.php"

def normalize(name):
    return re.sub(r"\s+", " ", name.strip().lower())

def wiki_lookup_gender(name):
    try:
        # Step 1: 검색
        params = {
            "action": "query", "list": "search", "srsearch": name,
            "format": "json", "srlimit": 1
        }
        r = requests.get(WIKI_API, params=params, timeout=10)
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]

        # Step 2: intro 텍스트 가져오기
        params = {
            "action": "query", "prop": "extracts", "explaintext": 1,
            "exintro": 1, "titles": title, "format": "json"
        }
        r = requests.get(WIKI_API, params=params, timeout=10)
        page = next(iter(r.json()["query"]["pages"].values()))
        text = page.get("extract", "").lower()

        # Heuristic
        if "actress" in text:
            return "female"
        if "actor" in text:
            return "male"
        he = text.count(" he ")
        she = text.count(" she ")
        they = text.count(" they ")
        if he > she and he >= 2:
            return "male"
        if she > he and she >= 2:
            return "female"
        if they >= 3:
            return "other"
        return None
    except Exception:
        return None

def main():
    names = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            names.append(row["positive"].strip())

    male, female, other, unknown = [], [], [], []

    Path(OUT_DIR).mkdir(exist_ok=True)

    for i, name in enumerate(names, 1):
        g = wiki_lookup_gender(name)
        if g == "male":
            male.append(name)
        elif g == "female":
            female.append(name)
        elif g == "other":
            other.append(name)
        else:
            unknown.append(name)

        if i % 50 == 0:
            print(f"{i}/{len(names)} done...")

        time.sleep(0.5)  # 요청 간격 (Wikipedia API 부담 줄이기)

    # Save
    for fname, lst in [
        ("positives_male.csv", male),
        ("positives_female.csv", female),
        ("positives_other.csv", other),
        ("positives_unknown.csv", unknown)
    ]:
        with open(Path(OUT_DIR)/fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["positive"])
            for n in lst:
                w.writerow([n])

    print("분류 완료!")
    print(f"male={len(male)}, female={len(female)}, other={len(other)}, unknown={len(unknown)}")

if __name__ == "__main__":
    main()