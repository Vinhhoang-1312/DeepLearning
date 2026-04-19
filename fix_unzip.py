import json

path = r'c:\Users\DELL\Desktop\Vinh Hoang\Master Program\Học sâu\Project\04_SwinTransformer_Classification.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = cell.get('source', [])
    if isinstance(src, list):
        new_src = []
        for line in src:
            if line == '    print(f"Giải nén: {z.name} ...")\n':
                new_src.append("    folder_name = z.with_suffix('')\n")
                new_src.append("    if not folder_name.exists() or not any(folder_name.glob('**/*.png')):\n")
                new_src.append('        print(f"Giải nén: {z.name} ...")\n')
            elif line == "    with zipfile.ZipFile(z, 'r') as zf:\n":
                new_src.append("        with zipfile.ZipFile(z, 'r') as zf:\n")
            elif line == "        zf.extractall(cls_dir_path)\n":
                new_src.append("            zf.extractall(cls_dir_path)\n")
            else:
                new_src.append(line)
        cell['source'] = new_src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Đã cập nhật logic giải nén trong notebook!")
