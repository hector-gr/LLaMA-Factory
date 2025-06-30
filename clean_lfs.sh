##############################################################################
# 0) fresh mirror clone (skip if you’re already inside one)                   #
##############################################################################
# GIT_LFS_SKIP_SMUDGE=1 git clone --mirror https://github.com/USER/REPO.git repo-clean.git && cd repo-clean.git

##############################################################################
# 1) list every blob whose content starts with the Git-LFS pointer header     #
#    — this is 100 % automatic, no hashes to type by hand.                    #
##############################################################################
git rev-list --objects --all | cut -d' ' -f1 | grep -v '^$' \
| xargs -r -n1 -P8 -I{} sh -c 'git cat-file -p {} 2>/dev/null | grep -q "^version https://git-lfs" && echo {}' \
| sort -u > pointer_blobs.txt             # <-- list of blobs to delete

##############################################################################
# 2) rewrite history: drop those blobs and prune commits that become empty   #
##############################################################################
pip install --quiet --user git-filter-repo && \
git filter-repo --force \
  --strip-blobs-with-ids pointer_blobs.txt \
  --prune-empty always

##############################################################################
# 3) delete the backup refs filter-repo created, then GC                     #
##############################################################################
git for-each-ref --format="%(refname)" refs/original/ \
| xargs -r -n1 git update-ref -d      #  -r  = “run only if there’s input”

# then finish the clean-up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

##############################################################################
# 4) sanity check – repo must be pointer-free                                 #
##############################################################################
git grep -I -n 'version https://git-lfs' $(git rev-list --all) || echo "✅ no LFS pointers"
git lfs ls-files || echo "✅ git lfs ls-files empty"

##############################################################################
# 5) push the cleaned repo (branches first, tags if you still want them)     #
##############################################################################
git remote set-url origin git@github.com:USER/REPO.git && \
git push --force --all origin && \
git push --force --tags origin        # ← omit if old tags aren’t needed