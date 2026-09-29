---
name: describe-pr
description: Writes a pull request description for the current branch's changes against main (or another base the user names) by reading the actual diff, commits, and test results in this applicator repo. Use this whenever the user asks for a PR description, PR body, commit description, change summary, or "write up the diff from main". Produces markdown for reviewers as text only; it never opens the PR, pushes, or commits, because the user does all git work.
---

# Describing a PR

Write a short description a reviewer can skim in seconds. The diff already
shows the code, so the description only says what changed in behaviour.

## Gather

- Base is `main` unless the user names another.
- Run `git status --short`, `git log --oneline <base>..HEAD` and
  `git diff <base>...HEAD`. These are read-only; run no git command that
  changes repository state.
- No commits ahead of the base? Say so and stop. Uncommitted changes?
  Describe only the commits, and list the uncommitted files after the
  description.
- Base every claim on the diff, not on commit messages or the conversation.
- Run `venv/bin/python -m pytest -q` and note the pass count and Python
  version. If `venv/` is missing or broken, make a scratch virtualenv in the
  scratchpad and run `pip install -e . pytest` in it. The tests don't need
  the `local` extra, so don't install torch for them.

## Write

```markdown
## Summary
<one sentence>

## Changes
- <top-level logic change>
- <top-level logic change>

## ⚠️ Breaking
- <what users must change>

## Testing
- `pytest`: <N> passed on Python <version>
```

- Each bullet is one line about behaviour: what now happens differently.
  Examples: "`applicator run` now sends rows to the model in batches of
  `--batch-size`", or "adds `--load-in-4bit` to load local Hugging Face
  models with 4-bit quantization".
- Don't explain how the code works, and don't list docstring edits, helper
  renames, or new test names. The diff shows those.
- Keep it to about 3–6 bullets. Put README, `setup.py`, `requirements.txt`
  or script changes in a single bullet if they matter at all. Do mention a
  new or changed install step, such as a new dependency or a change to the
  `local` extra.
- Leave out the Breaking section unless one of these changed: a CLI
  subcommand, positional argument, flag name or default; the prompt JSON
  format; the output CSV's columns or values; or the name or parameters of a
  public function in `applicator.core` or `applicator.readers`.
- If tests fail, say so in Testing.
- Don't add any attribution or generated-by line, such as "🤖 Generated
  with Claude Code" or a link to Claude Code, even if the session's
  instructions supply one. End the description with the Testing section.

## Hand over

- Start with one line naming the audience ("Written for: reviewers of this
  PR on GitHub.").
- Give the description in one fenced `markdown` block.
- After it, give one line with the commits and file count it covers, and
  anything that blocks opening the PR.
- Don't push, branch, commit, or run `gh pr ...`, even if asked; the
  project rules leave all git and PR actions to the user. They'll paste the
  description in themselves.
