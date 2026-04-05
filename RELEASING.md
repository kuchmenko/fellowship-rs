# Releasing agent-runtime

## Versioning

This project uses [Semantic Versioning](https://semver.org/):

- **Pre-1.0**: `0.MINOR.PATCH` — no stable API guarantees
  - Bump **MINOR** for breaking changes or significant features
  - Bump **PATCH** for bug fixes and small improvements
- **Post-1.0**: standard SemVer rules apply

## How releases work

This project uses [release-please](https://github.com/googleapis/release-please) for automated releases.

### Commit conventions

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add OpenAI provider          → bumps MINOR (0.1.0 → 0.2.0)
fix: handle empty tool response    → bumps PATCH (0.1.0 → 0.1.1)
feat!: redesign Tool trait         → bumps MINOR (breaking, pre-1.0)
chore: update dependencies         → no release
docs: update README                → no release
```

### Release flow

1. Push conventional commits to `main`
2. release-please automatically creates/updates a **Release PR** with:
   - Version bump in `Cargo.toml`
   - Updated `CHANGELOG.md`
3. **Merge the Release PR** when ready to ship
4. release-please creates a GitHub Release + git tag automatically

### Manual override

If you need to force a specific version or release out-of-band:

```
git tag vX.Y.Z
git push origin vX.Y.Z
```
