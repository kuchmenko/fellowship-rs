# Releasing agent-runtime

## Versioning

This project uses [Semantic Versioning](https://semver.org/):

- **Pre-1.0**: `0.MINOR.PATCH` — no stable API guarantees
  - Bump **MINOR** for breaking changes or significant features
  - Bump **PATCH** for bug fixes and small improvements
- **Post-1.0**: standard SemVer rules apply

## When to release

- After a meaningful batch of features or fixes lands on `master`
- After any breaking API change (bump minor)
- When a bug fix is urgent enough to warrant a standalone release

## Release checklist

1. **Verify CI is green** on `master`

2. **Update CHANGELOG.md**
   - Move items from `[Unreleased]` into a new `[X.Y.Z] - YYYY-MM-DD` section
   - Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security

3. **Bump version in `Cargo.toml`**
   ```
   version = "X.Y.Z"
   ```

4. **Commit the release**
   ```
   git add Cargo.toml Cargo.lock CHANGELOG.md
   git commit -m "release: vX.Y.Z"
   ```

5. **Tag and push**
   ```
   git tag vX.Y.Z
   git push && git push --tags
   ```

6. **Verify** the GitHub Release was created automatically from the tag

## Post-release

- Bump version in `Cargo.toml` to next dev version (e.g. `0.2.0` → `0.3.0`)
  if a new minor cycle is starting, or leave as-is for patch releases.
- Add a fresh `## [Unreleased]` section to CHANGELOG.md if it was consumed.
