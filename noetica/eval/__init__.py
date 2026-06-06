"""noetica.eval — golden-case evaluation for the structured-output API.

Runs on the core deps only (no torch). Two modes:

  * ``--check`` / ``--offline`` — validate the golden corpus against recorded
    fixtures with no live model. Deterministic; used as the CI regression gate.
  * live — call the running API for each case × model variant and print a
    pass/fail comparison table. Needs Ollama + the models pulled.

See :mod:`noetica.eval.run`.
"""
