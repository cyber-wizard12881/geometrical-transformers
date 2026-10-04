# jupytext --to notebook *.py

import json

# List the notebooks you want to combine in order
notebooks_to_merge = ['geo_generator.ipynb', 'hyperbolic.ipynb', 'spherical.ipynb', 'logexp.ipynb', 'parabolic.ipynb', 'geomapper.ipynb', 'transformer.ipynb', 'trainer.ipynb', 'tester.ipynb']

# Use the first notebook as a template to preserve metadata
with open(notebooks_to_merge[0], 'r', encoding='utf-8') as f:
    merged_notebook = json.load(f)

# Loop through the remaining notebooks and append their cells
for notebook_path in notebooks_to_merge[1:]:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        current_notebook = json.load(f)
        merged_notebook['cells'].extend(current_notebook['cells'])

# Save the combined content into a new file
with open('geometrical_transformers.ipynb', 'w', encoding='utf-8') as f:
    json.dump(merged_notebook, f, indent=1, ensure_ascii=False)

print("Notebooks merged successfully into 'geometrical_transformers.ipynb'!")