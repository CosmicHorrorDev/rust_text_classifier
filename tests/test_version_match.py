import toml

from lib import __version__
from tests.constants import REPO_DIR


# So the old packaging system allowed for some hacky stuff to automatically detect the
# version, but we're using `pyproject.toml` so the best I can think of is to test it
# with the test suite
def test_make_sure_versions_match() -> None:
    with (REPO_DIR / "pyproject.toml").open() as fh:
        contents = toml.load(fh)

    pyproject_version = contents["tool"]["poetry"]["version"]
    assert pyproject_version == __version__, "Version mismatch, update the versions!"
