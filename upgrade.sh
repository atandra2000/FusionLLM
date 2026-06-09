#!/usr/bin/env bash
# FusionLLM Upgrade Helper Scripts
# This script provides common commands for the upgrade process

set -e  # Exit on error

echo "=== FusionLLM Upgrade Helper ==="
echo "Running upgrades for FusionLLM repository..."
echo ""

# Check if we're in the correct directory
if [ ! -f "README.md" ]; then
    echo "Error: Must be run from FusionLLM root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  status       Show git status and modified files"
    echo "  diff         Show all changes (git diff)"
    echo "  backup       Backup current state (git stash)"
    echo "  unbackup     Restore backup (git stash pop)"
    echo "  review       Review all modified files"
    echo "  sync         Sync local changes to GitHub"
    echo "  test         Run smoke tests"
    echo "  docs-update  Update documentation"
    echo "  help         Show this help message"
    echo ""
}

# Function to show status
show_status() {
    echo "=== Git Status ==="
    git status
    echo ""
    echo "=== Modified Files ==="
    git diff --name-only
    echo ""
    echo "=== Untracked Files ==="
    git ls-files --others --exclude-standard
}

# Function to show diff
show_diff() {
    echo "=== All Changes (git diff) ==="
    git diff
    echo ""
    echo "=== Summary ==="
    git diff --stat
}

# Function to backup current state
backup() {
    echo "=== Creating backup (git stash) ==="
    git stash push -m "Backup before upgrade: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Backup created. To restore: ./upgrade.sh unbackup"
}

# Function to restore from backup
unbackup() {
    echo "=== Restoring from backup ==="
    git stash pop
    echo "Restored from backup."
}

# Function to review modified files
review() {
    echo "=== Reviewing Modified Files ==="
    
    MODIFIED=$(git diff --name-only)
    echo "Modified files:"
    for file in $MODIFIED; do
        echo "  - $file"
        echo "    Changes:"
        git diff --stat "$file"
    done
    
    UNTRACKED=$(git ls-files --others --exclude-standard | head -20)
    echo ""
    echo "Untracked files:"
    for file in $UNTRACKED; do
        echo "  - $file"
    done
}

# Function to sync with GitHub
sync_with_github() {
    echo "=== Syncing with GitHub ==="
    echo ""
    echo "Step 1: Check current branch"
    git branch
    echo ""
    
    echo "Step 2: Review changes"
    git diff --stat
    echo ""
    
    echo "Step 3: Stage all changes"
    git add .
    git status
    echo ""
    
    echo "Step 4: Commit changes"
    echo "Please provide a commit message:"
    read -p "Commit message: " commit_msg
    
    if [ -z "$commit_msg" ]; then
        commit_msg="chore: update FusionLLM with latest changes"
    fi
    
    git commit -m "$commit_msg"
    echo ""
    
    echo "Step 5: Push to GitHub"
    git push origin main
    echo ""
    
    echo "=== Sync complete! ==="
    echo ""
    echo "Next steps:"
    echo "1. Create a PR if working on a feature branch"
    echo "2. Update documentation"
    echo "3. Run tests to verify"
}

# Function to run smoke tests
run_tests() {
    echo "=== Running Smoke Tests ==="
    
    # Check if pytest is available
    if command -v pytest &> /dev/null; then
        echo "Testing with pytest..."
        pytest tests/ -v --tb=short -k "smoke or test_smoke"
    elif command -v python &> /dev/null; then
        echo "Testing with Python directly..."
        python tests/test_smoke.py
    else
        echo "Error: No test runner found (pytest or python)"
        exit 1
    fi
    
    echo ""
    echo "Smoke tests completed."
}

# Function to update documentation
update_docs() {
    echo "=== Documentation Updates ==="
    
    # Check docs directory
    if [ -d "docs" ]; then
        echo "Docs directory found."
        echo "Files in docs/:"
        ls -la docs/
        echo ""
    else
        echo "No docs/ directory found."
    fi
    
    # Check for new documentation files
    echo "Checking for untracked documentation..."
    UNTRACKED_DOCS=$(git ls-files --others --exclude-standard | grep -E "\.md$" || true)
    
    if [ -n "$UNTRACKED_DOCS" ]; then
        echo "Untracked documentation files:"
        echo "$UNTRACKED_DOCS"
        echo ""
        echo "To add these to version control, run:"
        echo "  git add $UNTRACKED_DOCS"
    else
        echo "No untracked documentation files."
    fi
}

# Command dispatcher
case "${1:-help}" in
    status)
        show_status
        ;;
    diff)
        show_diff
        ;;
    backup)
        backup
        ;;
    unbackup)
        unbackup
        ;;
    review)
        review
        ;;
    sync)
        sync_with_github
        ;;
    test)
        run_tests
        ;;
    docs|docs-update)
        update_docs
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac

echo ""
echo "=== Helper completed ==="
echo ""
