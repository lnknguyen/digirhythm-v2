# --- config ---
P1_URL="git@github.com:digitraceslab/digirhythm.git"
P2_URL="git@github.com:lnknguyen/Digirhythm.git"
SRC_DIR="figures"      # source subfolder in P1
DST_DIR="figures"      # destination path in P2

# --- fresh clone of P2 (or cd into an existing clone) ---
git clone "$P2_URL" P2 && cd P2

# Ensure you're on the default branch
git fetch origin
git checkout -B main origin/main || git checkout -B master origin/master

# If this repo was cloned via HTTPS earlier, switch origin to SSH:
git remote set-url origin "$P2_URL"

# Add P1 as an SSH remote
git remote add p1 "$P1_URL"
git fetch p1

# Detect P1 default branch (fallback to main)
P1_BRANCH=$(git ls-remote --symref "$P1_URL" HEAD | sed -n 's#^ref: refs/heads/\(.*\)\tHEAD#\1#p')
P1_BRANCH=${P1_BRANCH:-main}

# Create a split of just P1/$SRC_DIR (preserves subdir history)
SPLIT_COMMIT=$(git subtree split --prefix="$SRC_DIR" "p1/$P1_BRANCH")

# Bring that content into P2 at $DST_DIR
git subtree add --prefix="$DST_DIR" --squash . "$SPLIT_COMMIT"

# Push to your SSH origin
git push origin HEAD

