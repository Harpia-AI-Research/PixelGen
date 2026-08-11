# Publishing PixelGen to NPM

This guide explains how to publish PixelGen to the npm registry.

## Prerequisites

1. **npm account** - You need an npm account
   - Sign up at: https://www.npmjs.com/signup
   - Verify your email

2. **npm login** - Login to npm on your machine
   ```bash
   npm login
   ```
   Enter your:
   - Username
   - Password
   - Email
   - OTP (if 2FA enabled)

3. **Organization access** (if using scoped package)
   - Package name: `@harpia-ai/pixelgen`
   - You need access to the `harpia-ai` organization on npm
   - Or change package name to unscoped: `pixelgen-ml` or similar

## Pre-publish Checklist

- [ ] All tests pass (if you have tests)
- [ ] Build succeeds: `npm run build`
- [ ] Version updated in `package.json`
- [ ] CHANGELOG.md updated
- [ ] README.md is up to date
- [ ] LICENSE file present
- [ ] All dependencies are correct

## Verify Package Contents

Before publishing, check what will be included:

```bash
npm pack --dry-run
```

This shows:
- What files will be included
- Total package size
- Individual file sizes

Expected contents:
```
dist/             # Compiled JavaScript and .d.ts files
README.md         # Documentation
LICENSE           # License file
CHANGELOG.md      # Version history
package.json      # Package metadata
```

**Should NOT include:**
- `src/` - TypeScript source (only dist/)
- `examples/` - Example files
- `Documentation/` - Full documentation
- `node_modules/` - Dependencies
- `.git/` - Git history

## Publishing Steps

### 1. Login to npm

```bash
npm whoami
```

If not logged in:
```bash
npm login
```

### 2. Build the Package

```bash
npm run build
```

Verify dist/ was created with all files.

### 3. Test Local Installation (Optional)

```bash
# In another directory
npm install /path/to/PixelGen

# Try importing
node -e "const pg = require('@harpia-ai/pixelgen'); console.log(pg)"
```

### 4. Publish

For first-time publish:

```bash
npm publish --access public
```

**Note:** `--access public` is required for scoped packages (@harpia-ai/pixelgen)

For subsequent versions:

```bash
npm publish
```

### 5. Verify Publication

Check your package on npm:
- https://www.npmjs.com/package/@harpia-ai/pixelgen

Test installation:
```bash
mkdir test-pixelgen
cd test-pixelgen
npm install @harpia-ai/pixelgen
```

## Version Management

### Semantic Versioning

PixelGen follows [SemVer](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.0.1): Bug fixes, backwards compatible

### Updating Version

```bash
# Patch release (0.1.0 → 0.1.1)
npm version patch

# Minor release (0.1.0 → 0.2.0)
npm version minor

# Major release (0.1.0 → 1.0.0)
npm version major
```

This automatically:
- Updates package.json
- Creates a git commit
- Creates a git tag

Then:
```bash
git push && git push --tags
npm publish
```

## Troubleshooting

### Error: "Package name too similar to existing package"

Change package name in package.json to something unique:
- `pixelgen-ml`
- `pixelart-ml`
- `pixelgen-ai`

### Error: "You must be logged in to publish packages"

```bash
npm login
```

### Error: "You do not have permission to publish"

For scoped packages (@harpia-ai/pixelgen):
- Create the organization on npm: https://www.npmjs.com/org/create
- Or change to unscoped package name

### Error: "Package version already exists"

You're trying to publish a version that already exists. Update version:

```bash
npm version patch
npm publish
```

### Package is too large

Check size:
```bash
npm pack --dry-run
```

If over 10MB, check .npmignore to exclude unnecessary files.

## Post-Publish

### 1. Tag the Release on GitHub

```bash
git tag v0.1.0
git push origin v0.1.0
```

Or create a release on GitHub with release notes.

### 2. Update Documentation

Add installation instructions to README:

```markdown
## Installation

\`\`\`bash
npm install @harpia-ai/pixelgen
\`\`\`
```

### 3. Announce

- Tweet about it
- Post on relevant forums (Reddit, Discord servers)
- Update project homepage

## Maintenance

### Unpublishing

⚠️ **Warning:** Unpublishing is permanent and discouraged

Within 72 hours:
```bash
npm unpublish @harpia-ai/pixelgen@0.1.0
```

After 72 hours, unpublishing is not allowed (to protect the ecosystem).

### Deprecating

If a version is broken:
```bash
npm deprecate @harpia-ai/pixelgen@0.1.0 "This version has critical bugs. Please upgrade."
```

## CI/CD (Future)

For automated publishing via GitHub Actions:

1. Create npm token: https://www.npmjs.com/settings/tokens
2. Add to GitHub secrets as `NPM_TOKEN`
3. Create `.github/workflows/publish.yml`

---

## Quick Reference

```bash
# Login
npm login

# Build
npm run build

# Check contents
npm pack --dry-run

# Publish
npm publish --access public

# Update version
npm version patch

# Check publication
npm view @harpia-ai/pixelgen
```

---

**Ready to publish?** Follow the steps above! 🚀
