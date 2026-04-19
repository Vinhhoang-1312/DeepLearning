import json

path = r'c:\Users\DELL\Desktop\Vinh Hoang\Master Program\Học sâu\Project\04_SwinTransformer_Classification.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

count = 0
for cell in nb['cells']:
    src = cell.get('source', [])
    if isinstance(src, list):
        new_src = []
        for line in src:
            if 'window_size=4' in line:
                line = line.replace('window_size=4', 'window_size=5')
                count += 1
            # Remove indexing='ij' for older PyTorch compat
            if "indexing='ij'" in line:
                line = line.replace(", indexing='ij'", '')
                count += 1
            new_src.append(line)
        cell['source'] = new_src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Fixed {count} lines OK')
