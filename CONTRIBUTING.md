# Contributing

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q dimer monomer multicodes
python -m pytest
```

## Rules

- Never commit data, weights, outputs, logs, or credentials.
- Use English for file names, source comments, user-facing messages, and
  documentation.
- Keep experimental data paths configurable through environment variables or
  command-line arguments.
- Preserve image-level train/validation/test separation.
- Report model evaluation on an independent test set and label
  training-derived diagnostics explicitly.
- Describe measurement-derived randomness separately from deterministic
  pseudo-random expansion.
