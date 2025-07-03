import os, json, glob, random

# —— Configuration directories —— 
src1    = "./responses/MMLU-train"
src2    = "./responses/wmdp-train"
out_dir = "./responses/mmlu-wmdp-train"
os.makedirs(out_dir, exist_ok=True)

# —— Fixed random seed and file lists —— 
SEED = 42
random.seed(SEED)
files1 = sorted(glob.glob(os.path.join(src1, "*.json")))
files2 = sorted(glob.glob(os.path.join(src2, "*.json")))
common = [os.path.basename(p) for p in files1 if os.path.basename(p) in {os.path.basename(q) for q in files2}]

# —— Generate decision sequence only once —— 
n_per = 2900
total_len = n_per * 2
decisions = [random.random() < 0.5 for _ in range(total_len)]

def merge_fixed(l1, l2, decisions):
    i = j = d = 0
    merged = []
    while len(merged) < len(l1) + len(l2):
        if i >= len(l1):
            merged.append(l2[j]); j += 1
        elif j >= len(l2):
            merged.append(l1[i]); i += 1
        else:
            if decisions[d]:
                merged.append(l1[i]); i += 1
            else:
                merged.append(l2[j]); j += 1
            d += 1
    return merged

# —— Process each file with the same name —— 
for name in common:
    with open(os.path.join(src1, name),  encoding="utf-8") as f1: data1 = json.load(f1)
    with open(os.path.join(src2, name),  encoding="utf-8") as f2: data2 = json.load(f2)

    # Take first 2900 (preserve original order)
    chunk1 = data1[:n_per]
    chunk2 = data2[:n_per]

    # Use the same decisions list for interleaved merging
    merged = merge_fixed(chunk1, chunk2, decisions)

    # Write output
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fo:
        json.dump(merged, fo, ensure_ascii=False, indent=2)

print("Done. Each file uses exactly the same merging order.")