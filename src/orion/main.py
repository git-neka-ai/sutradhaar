# orion: Moved CLI entrypoint from editor.py into a dedicated module to match the console_script target and keep orion.py importable without side effects. Added a docstring explaining CLI usage and environment.

import pathlib
import sys

from .orion import Orion


# orion: Add a docstring describing usage, args, and environment variables consumed downstream.
def main() -> None:
    """
    Orion CLI entrypoint.

    Usage:
        orion [repo_root]

    Notes:
        - OPENAI_API_KEY and AI_MODEL must be set in the environment.
        - Optional: ORION_EXTERNAL_DIR and ORION_DEP_TTL_SEC influence dependency summaries.
        - If repo_root is not supplied, the current directory is used.
    """
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("Usage: orion [repo_root]")
        print("Environment: OPENAI_API_KEY, AI_MODEL")
        print("Optional External Dependencies (flat): ORION_EXTERNAL_DIR, ORION_DEP_TTL_SEC")
        return
    # orion: Accept an explicit repo_root unless argv[1] looks like a flag.
    repo_root = (
        pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 and not sys.argv[1].startswith("--") else pathlib.Path(".").resolve()
    )
    Orion(repo_root).run()


if __name__ == "__main__":
    main()
