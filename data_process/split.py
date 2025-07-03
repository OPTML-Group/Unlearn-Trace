import os, json, glob, random, re

# —— Configuration —— #
source_dir = "./responses/UltraChat"
train_dir  = "./responses/UltraChat-train"
eval_dir   = "./responses/UltraChat-eval"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir,  exist_ok=True)

TOTAL_TRAIN  = 5800
TOTAL_EVAL   = 360
TOTAL_SELECT = TOTAL_TRAIN + TOTAL_EVAL
SEED         = 12345

# —— Normalization function: remove punctuation, whitespace, and convert to lowercase —— #
_punc_space_re = re.compile(r"[\W_]+", flags=re.UNICODE)
def normalize_question(q: str) -> str:
    m = re.search(r"(Question:.*?\?)", q, flags=re.S)
    snippet = m.group(1) if m else q.replace("\n"," ")[:100]
    cleaned = _punc_space_re.sub("", snippet)
    return cleaned.lower()

# —— Simple "substring + length difference" matching —— #
def relaxed_equal(a: str, b: str, tol: int = 3) -> bool:
    # If the length difference between a and b is too large, return False directly
    if abs(len(a) - len(b)) > tol:
        return False
    # If a contains b or b contains a, consider them the same
    return (a in b) or (b in a)

# 1) Collect normalized key lists from all files
all_keys = []
per_file_keys = []
for path in glob.glob(os.path.join(source_dir, "*.json")):
    data = json.load(open(path, "r", encoding="utf-8"))
    keys = [ normalize_question(pair[0]["content"]) for pair in data ]
    per_file_keys.append(set(keys))
    all_keys.extend(keys)

# 2) Merge clusters based on relaxed_equal
clusters = []
for k in sorted(set(all_keys)):
    placed = False
    for cluster in clusters:
        if relaxed_equal(k, cluster[0], tol=3):
            cluster.append(k)
            placed = True
            break
    if not placed:
        clusters.append([k])

# Build member -> representative mapping
mapping = {}
for cluster in clusters:
    rep = cluster[0]
    for member in cluster:
        mapping[member] = rep

reps = list({mapping[k] for k in mapping})
print(f"After merging, there are {len(reps)} cluster representative keys")

# 3) Take strict intersection (at cluster representative level)
sets_of_reps = []
for keys in per_file_keys:
    sets_of_reps.append({ mapping[k] for k in keys if k in mapping })
common_reps = set.intersection(*sets_of_reps)
print(f"All files commonly contain {len(common_reps)} representative keys")

if len(common_reps) < TOTAL_SELECT:
    raise ValueError(f"Intersection only has {len(common_reps)} < {TOTAL_SELECT}, cannot sample")

# 4) Random sampling and splitting
rng = random.Random(SEED)
selected = rng.sample(sorted(common_reps), TOTAL_SELECT)
train_reps = set(selected[:TOTAL_TRAIN])
eval_reps  = set(selected[TOTAL_TRAIN:])

# 5) Split each file
for path in glob.glob(os.path.join(source_dir, "*.json")):
    fname = os.path.basename(path)
    data  = json.load(open(path, "r", encoding="utf-8"))
    qa_map = { normalize_question(pair[0]["content"]) : pair for pair in data }

    train_samples = []
    eval_samples  = []
    for orig_k, pair in qa_map.items():
        rep = mapping.get(orig_k)
        if rep in train_reps:
            train_samples.append(pair)
        elif rep in eval_reps:
            eval_samples.append(pair)

    with open(os.path.join(train_dir, fname), "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(eval_dir,  fname), "w", encoding="utf-8") as f:
        json.dump(eval_samples,  f, ensure_ascii=False, indent=2)

    print(f"{fname} → train: {len(train_samples)} samples, eval: {len(eval_samples)} samples")

print("✅ Completed: Applied relaxed matching rule with substring + length difference ≤3.")
