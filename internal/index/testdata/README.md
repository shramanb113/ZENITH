# Golden files

Fixtures for `golden_test.go`, which pins the on-disk contract in `FORMAT.md`.

- `golden-v5/golden.db` — a real format-5 (gob) index **written by the previous
  release's own library**: 20 documents with attributes and real vectors (a
  deterministic hashed bag-of-words embedder, `test:bag32`), then one delete and one
  replace. `expected.tsv` holds the search results that release returned for it
  (`Q` = plain query, `F` = query with an attribute filter), captured with
  `Limit(5)`.
- `golden-v6/` — the same corpus in the current format (manifest + one segment),
  produced by running `zenith migrate` on a copy of the v5 file.

Do **not** regenerate these to make a failing test pass. A failure means the format
or the search results changed: fix the code, or, for an intentional format change,
add a migration from the previous format and *add* new fixtures (keep the old ones).
