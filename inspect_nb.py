import json

nb_path = r"c:\Users\cs\OneDrive\Desktop\prompt\cnn-workshop-new.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(25, 31):
    cell = nb["cells"][idx]
    print(f"\n================ CELL {idx} ({cell['cell_type']}) ================")
    print("".join(cell["source"]))
