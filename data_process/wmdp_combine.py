import os, json, glob, random

bio_dir    = "./responses/wmdp-bio"
cyber_dir  = "./responses/wmdp-cyber"
output_dir = "./responses/wmdp"
os.makedirs(output_dir, exist_ok=True)

# --- 1) Compute max combined length, iterating in sorted order for determinism
lengths = []
bio_files = sorted(glob.glob(os.path.join(bio_dir, "*.json")))
for bio_path in bio_files:
    fname      = os.path.basename(bio_path)
    cyber_path = os.path.join(cyber_dir, fname)
    if not os.path.exists(cyber_path):
        continue
    bio_list   = json.load(open(bio_path,  encoding="utf-8"))
    cyber_list = json.load(open(cyber_path, encoding="utf-8"))
    lengths.append(len(bio_list) + len(cyber_list))

max_len = max(lengths)

# --- 2) Generate one fixed random-decisions list
random.seed(42)
decisions = [random.random() < 0.5 for _ in range(max_len)]

def merge_with_fixed_decisions(list1, list2, decisions):
    i = j = d = 0
    merged = []
    total = len(list1) + len(list2)
    while len(merged) < total:
        if i >= len(list1):
            merged.append(list2[j]); j += 1
        elif j >= len(list2):
            merged.append(list1[i]); i += 1
        else:
            if decisions[d]:
                merged.append(list1[i]); i += 1
            else:
                merged.append(list2[j]); j += 1
            d += 1
    return merged

# --- 3) Merge each file pair in sorted order
for bio_path in bio_files:
    fname      = os.path.basename(bio_path)
    cyber_path = os.path.join(cyber_dir, fname)
    if not os.path.exists(cyber_path):
        continue

    bio_list   = json.load(open(bio_path,   encoding="utf-8"))
    cyber_list = json.load(open(cyber_path, encoding="utf-8"))

    merged = merge_with_fixed_decisions(bio_list, cyber_list, decisions)
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(merged, fw, indent=2, ensure_ascii=False)

print("All files have been merged using the same decision sequence, ensuring consistent order.")
