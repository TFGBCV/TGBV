#!/usr/bin/env python3
"""PyInstaller entry point for the portable GBV Monitor bundle.

The imports below are not used directly: they exist so PyInstaller's analysis
sees all Streamlit modules the app will import at runtime (streamlit executes
app.py from source, so its own imports are invisible to the analysis of this
entry script). Real work happens in launcher.main().
"""
import multiprocessing  # noqa: F401  (freeze_support happens in launcher.main)
import sys

# --- app modules bundled & analysed -------------------------------------------
import alerts            # noqa: F401
import agent             # noqa: F401
import analyzer          # noqa: F401
import encryption        # noqa: F401
import eval              # noqa: F401
import jobs              # noqa: F401
import lexicon           # noqa: F401
import local_llm         # noqa: F401
import lmstudio          # noqa: F401
import memory            # noqa: F401
import monitor           # noqa: F401
import osint             # noqa: F401
import pdf_report        # noqa: F401
import report            # noqa: F401
import scraper           # noqa: F401
import storage           # noqa: F401
import vision            # noqa: F401

# lazy imports inside those modules — make them explicit for the analysis
import fpdf              # noqa: F401
import networkx          # noqa: F401
import PIL               # noqa: F401
import uharfbuzz         # noqa: F401
import altair            # noqa: F401
import pyarrow           # noqa: F401


def main() -> int:
    import launcher
    launcher.multiprocessing.freeze_support()
    return launcher.main()


if __name__ == "__main__":
    sys.exit(main())
