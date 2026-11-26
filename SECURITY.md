# Security Policy

Do not commit API keys, access tokens, AFM source data, annotations, model
weights, or unpublished result tables.

Before publishing a release:

1. Run `python -m pytest`.
2. Run a secret scanner over the full Git history, not only the current files.
3. Rotate any credential that has ever been committed.
4. Review reference slides and method figures for publication restrictions.

Report vulnerabilities privately to the repository maintainers.
