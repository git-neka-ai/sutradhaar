# orion: Moved CLI entrypoint from editor.py into a dedicated module to match the console_script target and keep orion.py importable without side effects.

import pathlib
import sys

from .orion import Orion


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("Usage: orion [repo_root]")
        print("Environment: OPENAI_API_KEY, AI_MODEL")
        print("Optional External Dependencies (flat): ORION_EXTERNAL_DIR, ORION_DEP_TTL_SEC")
        return
    repo_root = (
        pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 and not sys.argv[1].startswith("--") else pathlib.Path(".").resolve()
    )
    Orion(repo_root).run()


if __name__ == "__main__":
    main()
