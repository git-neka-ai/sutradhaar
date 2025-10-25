# orion: Moved CLI entrypoint from editor.py into a dedicated module. Updated to parse --external-dir/-e and remove ORION_EXTERNAL_DIR env usage; external dependencies are enabled only when the flag is provided.

import pathlib
import sys

from .orion import Orion


# orion: Update docstring to reflect CLI-driven external directory configuration and revised help text.
def main() -> None:
    """
    Orion CLI entrypoint.

    Usage:
        orion [--external-dir PATH|-e PATH] [repo_root]

    Notes:
        - OPENAI_API_KEY and AI_MODEL must be set in the environment.
        - Optional: ORION_DEP_TTL_SEC influences dependency summary TTL behavior.
        - If repo_root is not supplied, the current directory is used.

    Options:
        -e, --external-dir PATH   External Project Description directory (flat). When omitted, external dependency features are disabled.
    """
    args = sys.argv[1:]

    # Help handling (recognized anywhere in argv)
    if any(a in ("-h", "--help") for a in args):
        print("Usage: orion [--external-dir PATH|-e PATH] [repo_root]")
        print("Options:")
        print("  -e, --external-dir PATH   External Project Description directory (flat). When omitted, external deps are disabled.")
        print("Environment:")
        print("  OPENAI_API_KEY, AI_MODEL, ORION_DEP_TTL_SEC")
        return

    # orion: Implement minimal flag parsing for --external-dir/-e (supports --external-dir=PATH and -e PATH). The first non-flag argument is treated as repo_root.
    external_dir = None
    repo_root_arg = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-e":
            if i + 1 >= len(args):
                print("error: -e requires a PATH argument")
                return
            external_dir = args[i + 1]
            i += 2
            continue
        if a == "--external-dir":
            if i + 1 >= len(args):
                print("error: --external-dir requires a PATH argument")
                return
            external_dir = args[i + 1]
            i += 2
            continue
        if a.startswith("--external-dir="):
            external_dir = a.split("=", 1)[1]
            i += 1
            continue
        if a.startswith("-"):
            print(f"error: unknown option: {a}")
            return
        # First non-flag is repo_root
        if repo_root_arg is None:
            repo_root_arg = a
        i += 1

    repo_root = pathlib.Path(repo_root_arg).resolve() if repo_root_arg else pathlib.Path(".").resolve()
    Orion(repo_root, external_dir=external_dir).run()


if __name__ == "__main__":
    main()
