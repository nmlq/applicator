# Project rules

## Git: hands off

Never run any git command that changes repository state. The user reviews all
code and does every git action themselves, so they can validate each change.

- Do not stage, commit, amend, push, pull, merge, rebase, stash, reset,
  checkout, switch, tag, or create, rename or delete branches.
- Do not open, update or merge pull requests (`gh pr ...`).
- Leave all changes as uncommitted edits in the working tree and tell the user
  which files changed, so they can review, add and commit them.
- When a task would normally end with a commit or PR, stop at the edits and
  hand over: list the changed files and, if useful, a suggested commit message
  or PR description as text for the user to use.
