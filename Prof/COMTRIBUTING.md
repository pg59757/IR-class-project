# Contributing to Project

## Overview

This project follows a **fork-based independent workflow**. Each team works on their own fork of the repository.

**Important:**
- ❌ **Do NOT submit Pull Requests to the instructor's repository**
- ✅ **DO fork the repository and work on your team's fork**
- ✅ **DO pull updates when instructor announces them**
- ✅ Your team lead will give you access to your team's fork

---

## Table of Contents

- [Initial Setup (One-Time)](#initial-setup-one-time)
- [Daily Workflow](#daily-workflow)
- [Pulling Updates from Instructor](#pulling-updates-from-instructor)
- [Team Collaboration](#team-collaboration)
- [Submission for Grading](#submission-for-grading)
- [Troubleshooting](#troubleshooting)

---

## Initial Setup (One-Time)

### For Team Leads

**Step 1: Fork the Instructor's Repository**
https://github.com/uminho-biomed-pri/IR-class-project.git
1. Go to: `https://github.com/uminho-biomed-pri/IR-class-project.git`
2. Click **"Fork"** button (top-right)
3. **Owner:** Select your team account or personal account
4. **Repository name:** Keep as `IR-class-project` or use choose your own name
5. ✅ **Copy the `main` branch only**
6. Click **"Create fork"**

**You now have:** `https://github.com/YOUR_TEAM/IR-class-project`

**Step 2: Add Team Members as Collaborators**

1. Go to your fork: `https://github.com/YOUR_TEAM/IR-class-project`
2. **Settings → Collaborators → Add people**
3. Add all your team members with **Write** access
4. They'll receive email invitations

---

### For Team Members

**Step 1: Accept Invitation**

1. Check your email for the repository invitation
2. Click **"Accept invitation"**
3. You now have access to your team's fork

**Step 2: Clone Your Team's Fork**

```bash
# Clone YOUR TEAM'S fork (not the instructor's!)
git clone https://github.com/YOUR_TEAM/IR-class-project.git
cd IR-class-project
```

**Step 3: Add Instructor's Repo as Upstream**

```bash
# Add instructor's repository as "upstream"
git remote add upstream https://github.com/uminho-biomed-pri/IR-class-project.git

# Verify
git remote -v
```

**You should see:**
```
origin    https://github.com/YOUR_TEAM/IR-class-project.git (fetch)
origin    https://github.com/YOUR_TEAM/IR-class-project.git (push)
upstream  https://github.com/uminho-biomed-pri/IR-class-project.git (fetch)
upstream  https://github.com/uminho-biomed-pri/IR-class-project.git (push)
```

**Understanding:**
- **`origin`** = Your team's fork (where you push your work)
- **`upstream`** = Instructor's repository (where you pull updates from)

**Step 4: Set Up Development Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
pytest --version
```

---

## Daily Workflow

### Working on a New Feature

**Step 1: Create a Feature Branch**

```bash
# Start from your team's main or develop branch
git checkout main  # or 'develop' if your team uses it
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Examples:
# feature/add-search-endpoint
# feature/implement-docker
# bugfix/fix-timeout-error
# docs/update-api-docs
```

**Step 2: Make Your Changes**

```bash
# Work on your code
# Edit files, write tests, etc.

# Check what changed
git status
git diff
```

**Step 3: Commit Your Changes**

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "feat: implement search endpoint with filters"
```

**Good commit messages:**
```bash
✅ "feat: add TF-IDF search algorithm"
✅ "fix: resolve timeout error in scraper"
✅ "docs: update Docker setup instructions"
✅ "test: add unit tests for search module"

❌ "update"
❌ "fixes"
❌ "changes"
```

**Step 4: Push to Your Team's Fork**

```bash
# Push to YOUR TEAM's repository (origin)
git push origin feature/your-feature-name
```

**Step 5: Create Pull Request to Your Team's Fork**

1. Go to your team's fork on GitHub
2. You'll see: **"Compare & pull request"**
3. **Base repository:** YOUR_TEAM/IR-class-project
4. **Base branch:** `main` or `develop`
5. **Compare branch:** `feature/your-feature-name`
6. **Create pull request**

**Step 6: Team Review and Merge**

1. Team members review the PR
2. Team lead (or designated reviewer) approves
3. Merge to your team's `main` or `develop` branch
4. Delete the feature branch

---

## Pulling Updates from Instructor

**When the instructor announces updates are available:**

### Individual Team Members

```bash
# 1. Make sure you're on main
git checkout main

# 2. Fetch updates from instructor
git fetch upstream

# 3. Merge instructor's updates
git merge upstream/main

# 4. Push updates to your team's fork
git push origin main
```

**If you have uncommitted work:**
```bash
# Save your work temporarily
git stash

# Pull updates
git checkout main
git pull upstream main
git push origin main

# Go back to your branch
git checkout feature/your-feature

# Merge the updates into your branch
git merge main

# Restore your work
git stash pop
```

### Team Lead (Easier Method)

**Option 1: Sync via GitHub UI**

1. Go to your team's fork on GitHub
2. You'll see: **"This fork is X commits behind INSTRUCTOR/IR-class-project"**
3. Click **"Sync fork"** button
4. Click **"Update branch"**
5. Done! ✅

**Option 2: Command Line**

```bash
# Same as team members
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

**Announce to team:**
> "Updates pulled from instructor. Everyone please run `git pull origin main` to get the latest code."

---

## Team Collaboration

### Within Your Team's Fork

**You can work freely:**
- ✅ Create branches
- ✅ Submit PRs to your team's fork
- ✅ Review each other's code
- ✅ Merge to your `main` or `develop` branch
- ✅ Organize work however you want


---

## Submission for Grading

### When Instructor Requests Submission

**The instructor will specify submission method:**

**Option 1: GitHub Release (Most Common)**

```bash
# Tag your team's current state
git tag -a v1.0-submission -m "Sprint 1 submission"
git push origin v1.0-submission

# Create release on GitHub
1. Go to your fork → Releases → Create new release
2. Tag: v1.0-submission
3. Title: "Sprint 1 - Team DevOps Submission"
4. Description: Summary of work completed
5. Publish release
```

**Option 2: Share Repository Link**

Simply provide the instructor with:
```
Repository: https://github.com/YOUR_TEAM/IR-class-project
Branch: main
Commit: abc1234 (or tag: v1.0-submission)
Date: 2024-02-19
```

**Option 3: Export and Submit**

```bash
# Create a zip archive
git archive --format=zip --output=team-devops-sprint1.zip main

# Submit the zip file via course platform
```

**What the Instructor Will Review:**

The instructor will:
1. Visit your team's fork
2. Review commits, PRs, code quality
3. Run tests on your code
4. Check documentation
5. Grade based on rubric

---

## Grading Criteria

The instructor will evaluate **on your team's fork**:

### Code Quality (40%)
- ✅ Working functionality
- ✅ Code organization
- ✅ Follows best practices
- ✅ Proper error handling

### Testing (20%)
- ✅ Unit tests present
- ✅ Tests pass
- ✅ Good coverage
- ✅ Edge cases tested

### Documentation (20%)
- ✅ README updated
- ✅ Code comments
- ✅ API documentation
- ✅ Setup instructions

### Collaboration (20%)
- ✅ Meaningful commits
- ✅ PR reviews
- ✅ Team participation
- ✅ Git workflow followed

---

## Troubleshooting

### Problem: Can't Push to Instructor's Repository

**This is expected!** You should **only** push to your team's fork.

```bash
# ❌ WRONG - Don't do this
git push upstream main

# ✅ CORRECT - Push to your team's fork
git push origin main
```

### Problem: Fork is Behind Instructor's Repo

**Solution: Pull updates from upstream**

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

**Or use GitHub's "Sync fork" button** (easier!)

### Problem: Merge Conflicts When Pulling Updates

```bash
# When you run: git merge upstream/main
# Git shows: CONFLICT (content) in file.py

# 1. Open the conflicted file
# 2. Look for conflict markers:
<<<<<<< HEAD
your team's code
=======
instructor's code
>>>>>>> upstream/main

# 3. Decide what to keep
# 4. Remove the markers
# 5. Save the file

# 6. Mark as resolved
git add file.py

# 7. Complete the merge
git commit -m "Merge updates from instructor, resolve conflicts"

# 8. Push to your fork
git push origin main
```

### Problem: Team Member Can't Access Fork

**Solution: Team lead needs to add them**

1. Fork Settings → Collaborators
2. Add team member
3. They accept invitation email

### Problem: Want to Show Instructor Progress Before Deadline

**Solution: Give instructor read access**

1. Your fork → Settings → Collaborators
2. Add instructor with **Read** access
3. They can view but not modify

**Or:** Make your fork public 
- Settings → General → Change visibility → Public

---

## Best Practices

### ✅ DO

- **Pull updates regularly** from instructor (weekly)
- **Communicate with team** before major changes
- **Create small, focused commits**
- **Write clear commit messages**
- **Review teammates' PRs promptly**
- **Keep main branch stable**
- **Test before pushing**
- **Document your work**

### ❌ DON'T

- **Don't push directly to main** (use PRs)
- **Don't submit PRs to instructor's repo**
- **Don't ignore instructor updates**
- **Don't commit secrets or passwords**
- **Don't force push to shared branches**
- **Don't commit generated files** (pycache, node_modules)
- **Don't work on main branch directly**

---

## Communication Flow

```
Instructor's Repository
    ↓ (announcements, updates)
    
Your Team's Fork
    ↓ (work, collaborate)
    
Team Members
    ↓ (individual features)
    
Pull Requests (within team)
    ↓ (review, merge)
    
Team's Main Branch
    ↓ (tag for submission)
    
Instructor Reviews Your Fork
```

**Instructor never receives PRs from you!**  
**Instructor reviews your work on your fork!**

---

## Instructor Updates - What to Expect

The instructor may push updates for:

### Bug Fixes
```
Instructor: "Bug fix pushed to main. Please pull updates."
You: git pull upstream main
```

### New Features/Requirements
```
Instructor: "Added new API contract in docs/. Pull updates."
You: git pull upstream main
```

### Starter Code
```
Instructor: "Added starter code for Sprint 2. Pull updates."
You: git pull upstream main
```

### Test Updates
```
Instructor: "Updated test suite. Pull updates."
You: git pull upstream main
```

**When you see these announcements:**
1. Pull immediately: `git pull upstream main`
2. Resolve any conflicts
3. Push to your fork: `git push origin main`
4. Notify your team
5. Continue work

---

## Team Organization Tips

### Recommended Team Structure

**Team Lead:**
- Manages repository access
- Reviews and merges PRs
- Coordinates with instructor
- Ensures main branch stability

**Developers:**
- Work on feature branches
- Submit PRs to team fork
- Review each other's code
- Help teammates

**Quality Assurance:**
- Review PRs for quality
- Run tests
- Check documentation
- Verify requirements

### Communication Channels

- **GitHub Issues** - Track tasks, bugs
- **GitHub Discussions** - Team decisions, Q&A
- **PR Comments** - Code-specific feedback
- **Team Meetings** - Weekly sync-ups
- **Slack/Discord** - Quick questions

---

## Quick Reference

### Common Commands

```bash
# Setup (one time)
git clone https://github.com/YOUR_TEAM/IR-class-project.git
cd IR-class-project
git remote add upstream https://github.com/uminho-biomed-pri/IR-class-project.git

# Start new feature
git checkout main
git pull origin main
git checkout -b feature/my-feature

# Save work
git add .
git commit -m "feat: descriptive message"
git push origin feature/my-feature

# Pull instructor updates
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Update your feature branch with latest
git checkout feature/my-feature
git merge main
```

### Workflow Diagram

```
Instructor Repository (upstream)
        ↓ (you pull updates)
    
Your Team's Fork (origin)
        ↓ (you push work)
    
Your Local Repository
        ↓ (you work here)
    
Feature Branches
        ↓ (PRs within team)
    
Team's Main Branch
        ↓ (instructor reviews)
    
Grading
```

---

## Summary

**Key Points:**
1. ✅ Fork the instructor's repository (team lead does this once)
2. ✅ Work on your team's fork
3. ✅ Pull updates from instructor when announced
4. ❌ Never submit PRs to instructor's repository
5. ✅ Submit by tagging releases or sharing your fork link
6. ✅ Instructor reviews your work on your fork

**Remember:**
- Your team's fork is your workspace
- Instructor's repository is read-only (for pulling updates)
- Collaborate freely within your team
- Pull instructor updates regularly
- Submit when requested via tags/releases

---

**Questions?**

- Check existing GitHub Issues on instructor's repo
- Ask in course Discussions
- Contact your team lead
- Attend class hours

**Happy coding! 🚀**
