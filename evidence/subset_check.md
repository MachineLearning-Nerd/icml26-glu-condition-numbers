# Judged-to-candidate subset check

Judged revision:
`DineshAI/w0JhOFWPJl@a9288ab5d4762defa1c9f49f8a58aa27377f5d7d`.

Procedure:

1. Download the exact judged revision into a fresh empty directory.
2. Overlay only the prepared text allowlist.
3. Compare sorted relative file paths with `comm -23`.
4. Independently hash the historical evidence page.

Result:

- Paths present in judged but absent in candidate: `0`.
- The judged 13-file set is a subset of the candidate file set.
- `pages/overview/page.md` judged SHA-256:
  `60dcb1e34574353e822b4df1e7be1fb570e1eaadb894b8f82d1408dcafaeb2d6`.
- `pages/overview/page.md` candidate SHA-256:
  `60dcb1e34574353e822b4df1e7be1fb570e1eaadb894b8f82d1408dcafaeb2d6`.

The navigation/configuration files `README.md`, `logbook.json`, and
`pages/index.md` are additively updated to put current verification first.
The historical overview itself is unchanged and labeled exactly
**Historical rejected baseline**.
