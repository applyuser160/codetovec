import os

import nox


@nox.session(venv_backend="uv", python=["3.13"], tags=["test"])
def test(session: nox.Session) -> None:
    session.run("uv", "sync", "--dev", "--active")
    session.env["PYTHONPATH"] = os.path.abspath(".")
    args = session.posargs or ["tests"]
    session.run("pytest", *args)


@nox.session(venv_backend="uv", python=["3.13"], tags=["coverage"])
def test_coverage(session: nox.Session) -> None:
    session.run("uv", "sync", "--dev", "--active")
    session.env["PYTHONPATH"] = os.path.abspath(".")
    session.run("pytest", "--cov=src", "--cov-report=json:coverage.json")
