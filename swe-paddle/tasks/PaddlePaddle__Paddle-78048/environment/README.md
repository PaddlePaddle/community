# Environment

This task is Python-only and does not require a Paddle source rebuild.

Recommended verification environment:

- Windows or Linux
- Python 3.11
- pytest
- NumPy
- an installed Paddle wheel providing the native CPU runtime

The verifier loads the target Python module from the Base/Solution worktree, so the behavior under test comes from the checked-out source rather than from the installed wheel's copy of `manipulation.py`.
