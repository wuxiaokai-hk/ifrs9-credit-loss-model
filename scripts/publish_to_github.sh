#!/usr/bin/env bash
# Create the GitHub repository and push this project.
#   ./scripts/publish_to_github.sh <github-username> [public|private]
# Requires git and the GitHub CLI (`gh auth login` once). Then deploy the app at https://share.streamlit.io.
set -euo pipefail
USER_NAME="${1:?usage: publish_to_github.sh <github-username> [public|private]}"
VIS="${2:-public}"
REPO="ifrs9-credit-loss-model"
cd "$(dirname "$0")/.."
sed -i.bak "s/GITHUB_USER/${USER_NAME}/g" README.md && rm -f README.md.bak
git init -q -b main 2>/dev/null || git init -q
git add -A
git commit -q -m "IFRS 9 credit-loss scenario model v1.0.0" || true
gh repo create "${USER_NAME}/${REPO}" --"${VIS}" --source . --remote origin --push \
  --description "Scenario credit-loss projection and IFRS 9 ECL engine, with an interactive Streamlit app"
echo "Pushed: https://github.com/${USER_NAME}/${REPO}"
echo "Deploy the app: https://share.streamlit.io -> New app -> ${USER_NAME}/${REPO}, branch main, file streamlit_app.py"
