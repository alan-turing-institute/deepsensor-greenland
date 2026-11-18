"""Local demo site with a live, Arraylake-backed data dashboard.

Serves the static pages under the repo's ``demo/`` directory plus the JSON/PNG
endpoints the Data page's dashboard reads, streaming data on demand from an
Earthmover/Arraylake repo. Run with ``uv run python -m ds_greenland.demo``.

Adapted from the ``acocac-demo-dashboard`` Hugging Face Space.
"""
