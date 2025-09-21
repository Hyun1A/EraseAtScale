import torch
import os
import numpy as np
import yaml, csv
import json


concept_path = "/home/ldlqudgus756/EraseAtScale/EAS/EAS_sd1.4/pm_cache"

concepts = os.listdir(concept_path)

concepts = [con for con in concepts if "similar" in con]

print(concepts)

results = {}

for con in concepts:
    data = torch.load(f"{concept_path}/{con}", map_location=torch.device("cpu"))
    target = " ".join(con.split(".")[0].split("_")[1:])
    mappings = [s.strip() for s in data.keys()]
    print(f"{target}: {mappings}")
    results[target] = mappings

output_path = "./notes/concept_mappings.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Saved to {output_path}")