# Project Page — Conflict-Aware Active Perception in 3DGS

Set up to be served from `/docs` on `main` of <https://github.com/sircesoc/Conflict_Aware_Active_Perception>.

Final URL: `https://sircesoc.github.io/Conflict_Aware_Active_Perception/`

## Deploy

Copy this `docs/` folder into the root of your local clone, then:

```bash
git add docs/
git commit -m "Add project page"
git push origin main
```

Then on GitHub: **Settings → Pages → Source: Deploy from a branch → Branch: main, /docs → Save**.

## Local preview

```bash
cd docs && python3 -m http.server 8000
```

## File structure

```
docs/
├── index.html
└── static/
    ├── css/index.css
    ├── pdf/paper.pdf
    └── videos/teaser.mp4
```

## Still to fill in

- arXiv link in `index.html`
- Author profile links
- BibTeX citation key

Template: adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io) (CC BY-SA 4.0).
