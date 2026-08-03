---
name: commit
description: Create a git commit with staged changes. Claude reviews the diff, writes a clear commit message, and commits to the current branch.
disable-model-invocation: true
---

Create a git commit with the following workflow:

1. Run `git status` to see staged and unstaged changes
2. Run `git diff --cached` to review staged changes
3. Analyze the changes and write a clear commit message following conventional commits
4. Run `git commit -m "message"` with the message
5. Run `git status` to confirm the commit succeeded

Commit message guidelines:
- Start with type: feat, fix, refactor, docs, test, chore
- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Add blank line then detailed explanation if needed
- Example: "feat: add user authentication with JWT"

Do not commit if there are no staged changes. Ask the user to stage changes first.