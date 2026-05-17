# BFCL v3 `simple` — provenance

Vendored from the Berkeley Function-Calling Leaderboard (Gorilla project).

## Retrieval

- Retrieval date: 2026-05-17
- Upstream commit at time of retrieval: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- Last upstream commit that touched the v3 `simple` files: `c15b2a151662` (2025-06-09). Files were renamed to `BFCL_v4_*` in that commit; the v3 paths used below resolve via `c15b2a151662~1`.

## Source files (merged into one vendored file)

1. Prompts + tool schemas:
   `https://raw.githubusercontent.com/ShishirPatil/gorilla/c15b2a151662~1/berkeley-function-call-leaderboard/data/BFCL_v3_simple.json`
2. Gold answers (`possible_answer`):
   `https://raw.githubusercontent.com/ShishirPatil/gorilla/c15b2a151662~1/berkeley-function-call-leaderboard/data/possible_answer/BFCL_v3_simple.json`

Each line of the vendored `BFCL_v3_simple.json` is the upstream prompt record with a `possible_answer` key spliced in from the matching `ground_truth` entry. No other transformations.

Record count: 400. The file is JSON Lines (one JSON object per line) despite the `.json` extension; this matches BFCL's upstream convention.

## License

The BFCL data is distributed by the Gorilla project under the Apache License 2.0. See `https://github.com/ShishirPatil/gorilla/blob/main/LICENSE` for the full text.
