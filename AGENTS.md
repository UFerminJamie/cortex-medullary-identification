# AGENTS.md

## Project
This project classifies kidney whole-slide image (WSI) tiles as cortical or medullary using a GPU-accelerated support vector machine (SVM) from RAPIDS cuML. It maps tile-level predictions back to slide coordinates to produce spatial prediction heatmaps.

## Environment and platform
- The primary workflow runs on UF HiPerGator.
- The Conda environment is `/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu`.
- The workflow uses TRIDENT for WSI processing and tile embedding extraction, plus CuPy and RAPIDS cuML for GPU-based SVM operations.
- Do not assume GPU access is available locally. Check the project setup scripts and scheduler requirements before running GPU commands.
- Do not create or activate a different environment unless the user requests it.

## Project workflow
The workflow includes:
1. Processing WSIs and extracting tile embeddings.
2. Assigning tile labels from annotations.
3. Creating slide-level training, validation, and test splits.
4. Training the GPU SVM.
5. Running inference and generating spatial heatmaps.

Before changing a workflow stage, trace its inputs and outputs through the relevant scripts and documentation. Preserve established data formats and slide-level split behavior.
The active Python modules and optional H5 QA utilities live in `scripts/`; the root shell launchers remain the workflow entrypoints.

## Before making changes
- Read the repository’s `README.md` before making changes. Also read relevant documentation for the files or workflow being modified.
- Confirm the repository is a Git checkout. Inspect the current branch, working tree status, and diff before editing.
- Preserve all pre-existing user changes. Do not reset, clean, or overwrite them.
- Follow the existing project structure, naming, and code style.
- Keep changes focused on the requested task; do not reformat or modify unrelated files.
- Do not add dependencies unless necessary. If adding one, explain why and update the appropriate dependency documentation.
- Keep machine-specific paths configurable where the project already supports configuration. Do not silently replace established HiPerGator paths.

## Finding and reading files
- Use `rg --files` to find files and `rg` to search their contents.
- Do not use `cat` to read files.
- Inspect only relevant sections of large files; use commands such as `sed -n '1,160p' path/to/file`.
- Avoid dumping large files, directory trees, or data files when a targeted search or summary will work.

## Code quality
- Write clear, maintainable code that follows the project’s existing style.
- Keep implementations focused; avoid unnecessary code, duplication, and abstractions.
- Before adding a function or utility, search the repository for an existing implementation and reuse or extend it when appropriate.
- Do not add a new dependency or helper for behavior the project already provides.

## Pre-change backup
- Before editing files, create a timestamped backup of the current codebase under `${TMPDIR:-/tmp}`.
- Include tracked source and configuration files, plus untracked files that may be affected.
- Exclude `.git`, environments, dependencies, caches, large datasets, model checkpoints, generated outputs, and build artifacts. Check `.gitignore` and project documentation for project-specific exclusions. Examples include `.venv`, `venv`, `node_modules`, `data`, `datasets`, `outputs`, `checkpoints`, and `build`.
- Do not include credentials, tokens, or other secrets in the backup.
- Record the backup path and exclusions in the completion summary.
- If asked to restore the pre-change state, restore from that task’s backup. Preserve changes made after the backup unless the user requests a full restore.
- Do not delete an existing backup unless the user asks.

## Data and generated files
- Treat WSIs, annotations, embeddings, model checkpoints, predictions, and heatmaps as research data or generated artifacts.
- Do not delete, move, overwrite, or regenerate existing data unless the task explicitly requires it.
- Do not add large datasets, model weights, or generated outputs to Git unless explicitly requested.
- Never expose credentials, tokens, or sensitive data in code, logs, documentation, or command output.

## Editing and documentation
- Preserve existing file encoding and line endings.
- Update documentation when setup, commands, or behavior changes.
- For substantial changes, update `README.md` when the change affects documented behavior, setup, commands, or usage. Skip the update when the README is not relevant.
- Prefer project-defined scripts and commands over introducing new tooling.

## Verification
- Run the most relevant project-defined tests, lint checks, or type checks for the change.
- For GPU-dependent changes, use the documented HiPerGator setup and job workflow when available. Do not claim GPU verification if it was not run on a GPU.
- Do not claim a test, training run, or inference run succeeded unless you ran it and confirmed the result.
- If a check cannot be run, state why and what remains unverified.

## Git and external actions
- Make changes on the current branch unless the user requests a separate branch or worktree.
- When asked to revert your changes, use the relevant Git diff or that task’s backup to restore only those changes. Do not undo unrelated user work.
- Do not commit, push, publish, or deploy unless the user explicitly requests it.
- Ask before destructive actions or actions that affect external systems.

## Git command handoff
- At the end of every task that changes project files, provide copy-ready, ordered commands to create a new branch, stage the intended files, review and commit them, push the branch, open a GitHub pull request, and merge that pull request.
- Tailor the commands to the current branch, remote, and changed files. Base the pull request on the branch used to create the new branch unless the user specifies another base. State any required tools or authentication that are unavailable locally.
- Provide these commands for the user to run. Do not run branch creation, staging, commits, pushes, pull-request creation, or merges unless the user specifically asks you to execute them on their behalf.

## Completion summary
When finished, report:
- What changed and why.
- Which files changed.
- Checks run and their results.
- Each backup directory created, with its full path.
- Total disk space used by the project backups, for example, using `du -sh`.
- Any known limitations or follow-up work.
