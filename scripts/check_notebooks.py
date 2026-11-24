"""Execute all notebooks in fresh kernels using the active Python environment."""

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        # The kernelspec is in memory: no global Jupyter installation or user config is changed.
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client.km = manager
        with tempfile.TemporaryDirectory(prefix="creditguard-notebook-") as cache:
            client.execute(env={**os.environ, "MPLCONFIGDIR": cache})
        nbformat.write(notebook, path)
        print(f"Executed {path.name}")


if __name__ == "__main__":
    main()
