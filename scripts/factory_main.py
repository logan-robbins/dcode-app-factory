"""Run the AI software product factory end-to-end.

This script is a thin shim; the real implementation lives in
dcode_app_factory.__main__ so it is reachable via `dcode` console script.
"""

from dcode_app_factory.__main__ import load_raw_request, main

if __name__ == "__main__":
    raise SystemExit(main())
