How to export the Mermaid diagram to PNG / SVG

This folder contains `architecture.mmd` (Mermaid markup).

Recommended options to export to PNG or SVG:

1) Using `mmdc` (Mermaid CLI) via npm (Node.js required):

```bash
# install once
npm install -g @mermaid-js/mermaid-cli

# export SVG
mmdc -i docs/figures/architecture.mmd -o docs/figures/architecture.svg

# export PNG (specify width/height if needed)
mmdc -i docs/figures/architecture.mmd -o docs/figures/architecture.png
```

2) Using npx (no global install):

```bash
npx @mermaid-js/mermaid-cli -i docs/figures/architecture.mmd -o docs/figures/architecture.svg
```

3) From VS Code: open `docs/architecture.md` or `docs/figures/architecture.mmd` and use a Mermaid preview extension, then export manually via the extension UI.

If you want, I can run the `mmdc` command here to generate the SVG/PNG — confirm and I will execute it.