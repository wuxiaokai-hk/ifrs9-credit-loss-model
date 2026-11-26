# Publishing and hosting

## 1. GitHub

```bash
gh auth login                                   # once
./scripts/publish_to_github.sh <your-github-username> public     # or private
```

The script fills your username into the README badges, makes the first commit and pushes. CI (lint, 31 tests, a CLI
run) starts on the first push. Without the GitHub CLI: create an empty repository named `ifrs9-credit-loss-model` on
github.com, then `git init -b main && git add -A && git commit -m "v1.0.0" && git remote add origin <url> && git push -u origin main`.

## 2. Live app on Streamlit Community Cloud (free)

1. Sign in at https://share.streamlit.io with GitHub.
2. *Create app* → *Deploy a public app from GitHub* → repository `<you>/ifrs9-credit-loss-model`, branch `main`,
   main file `streamlit_app.py`. Optionally choose a custom subdomain, e.g. `ifrs9-credit-loss`.
3. Deploy. The first build installs the package from `requirements.txt` (a few minutes). The app opens on the
   shipped EQB results, so visitors see numbers immediately.

Visitors' runs are written to a private per-session folder (`outputs/app/`), and the shipped configurations are
read-only in the app, so no visitor can change what the next one sees. The container's disk is reset on each deploy.

## 3. Docker (any host)

```bash
docker build -t creditloss .
docker run -p 8501:8501 creditloss          # http://localhost:8501
```

## 4. Before you share it

- Put the app link and the repository link in the README's first lines and on your resume.
- A public repository is visible to anyone, including colleagues. If you want the job search to stay low-profile,
  publish it as `private` and decide later; Streamlit Cloud's options for apps from private repositories change over
  time, so check its current terms before deploying one.
