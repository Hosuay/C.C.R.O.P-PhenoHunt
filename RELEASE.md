# Release Process for C.C.R.O.P-PhenoHunt

## Sacred Geometry Release Strategy

Following sacred numerology principles:
- **Major versions**: 3, 6, 9, 27, 33, 369
- **Minor versions**: Aligned with feature counts
- **Patches**: Standard semantic versioning

Current version: **3.0.0**

---

## Creating a New Release

### 1. Pre-Release Checklist (9 items)

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code quality checks pass (`flake8 src/`, `black --check src/`)
- [ ] Documentation updated (README.md, CHANGELOG.md)
- [ ] Version number bumped in appropriate files
- [ ] Docker images build successfully
- [ ] Notebooks execute without errors
- [ ] Benchmarks documented in BENCHMARKS.md
- [ ] Sacred geometry constants verified
- [ ] Git branch is clean and up-to-date

### 2. Version Numbering

```bash
# Determine next version based on changes
MAJOR.MINOR.PATCH

# Sacred geometry major versions:
# 3.x.x - Initial research suite
# 6.x.x - Multi-language integration (Python, C++, Rust)
# 9.x.x - Production-ready platform
# 27.x.x - Federated learning integration
# 369.x.x - Ultimate vision release
```

### 3. Update Version Files

Update version in the following files:

```bash
# Python package
src/phenohunter_scientific.py:__version__ = "X.Y.Z"

# README.md
README.md:version-X.Y.Z

# CHANGELOG.md (add new section)
CHANGELOG.md
```

### 4. Create Release Tag

```bash
# Create annotated tag with sacred geometry message
git tag -a vX.Y.Z -m "[HARMONIC-369] RELEASE: Version X.Y.Z

Sacred Geometry Alignment: <numerology>
Major Features:
- Feature 1
- Feature 2
- Feature 3

🤖 Generated with Claude Code
"

# Push tag
git push origin vX.Y.Z
```

### 5. Build Docker Images

```bash
# Build all images with version tag
docker build -f docker/phenomics.Dockerfile -t crop-phenohunt/phenomics:vX.Y.Z .

# Tag as latest if stable
docker tag crop-phenohunt/phenomics:vX.Y.Z crop-phenohunt/phenomics:latest

# Push to registry
docker push crop-phenohunt/phenomics:vX.Y.Z
docker push crop-phenohunt/phenomics:latest
```

### 6. Create Zenodo Snapshot

1. Go to https://zenodo.org/
2. Connect to GitHub repository
3. Create new release on GitHub
4. Zenodo automatically creates a DOI
5. Update CITATION.cff with new DOI

### 7. Generate Release Notes

Create GitHub release with the following structure:

```markdown
# C.C.R.O.P-PhenoHunt vX.Y.Z

## 🔮 Sacred Geometry Alignment
- Harmonic seed: 369
- Feature dimensions: 27
- Pipeline stages: 3

## ✨ New Features
- Feature 1 description
- Feature 2 description
- Feature 3 description

## 🔧 Improvements
- Improvement 1
- Improvement 2

## 🐛 Bug Fixes
- Fix 1
- Fix 2

## 📊 Benchmarks
See [BENCHMARKS.md](BENCHMARKS.md) for performance metrics.

## 📚 Documentation
- Updated README with new features
- Added X new notebooks
- Enhanced API documentation

## 🐳 Docker Images
\`\`\`bash
docker pull crop-phenohunt/phenomics:vX.Y.Z
\`\`\`

## 📦 Installation
\`\`\`bash
pip install git+https://github.com/Hosuay/C.C.R.O.P-PhenoHunt.git@vX.Y.Z
\`\`\`

## 🔗 Resources
- **DOI**: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
- **Documentation**: [README.md](README.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

**Full Changelog**: https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/compare/vX.Y.Z-1...vX.Y.Z
```

### 8. Announce Release

- [ ] Update repository README badges
- [ ] Notify collaborators
- [ ] Post on relevant forums/communities
- [ ] Update project website (if applicable)

---

## Hotfix Releases

For critical bug fixes:

```bash
# Create hotfix branch
git checkout -b hotfix/vX.Y.Z+1 vX.Y.Z

# Make fixes
git commit -m "[HARMONIC-XXX] FIX: Critical bug description"

# Tag and release
git tag -a vX.Y.Z+1 -m "Hotfix: Bug description"
git push origin vX.Y.Z+1
```

---

## Release Artifacts

Each release should include:

1. **Source Code Archive** (ZIP, TAR.GZ)
2. **Docker Images** (all variants)
3. **Documentation** (HTML, PDF)
4. **Example Datasets** (small samples)
5. **CITATION.cff** (with DOI)
6. **SBOM** (Software Bill of Materials)

---

## Zenodo Integration

### Automatic DOI Generation

1. **Enable Zenodo Integration**
   - Go to https://zenodo.org/account/settings/github/
   - Connect GitHub account
   - Enable C.C.R.O.P-PhenoHunt repository

2. **Create Release on GitHub**
   - Zenodo automatically detects new releases
   - Generates DOI within minutes

3. **Update CITATION.cff**
   ```yaml
   cff-version: 1.2.0
   title: "C.C.R.O.P-PhenoHunt"
   version: X.Y.Z
   doi: 10.5281/zenodo.XXXXXXX
   date-released: 2025-XX-XX
   ```

### Manual Zenodo Upload

If automatic integration fails:

1. Create new upload on Zenodo
2. Upload release archive
3. Fill metadata:
   - Title: "C.C.R.O.P-PhenoHunt vX.Y.Z"
   - Authors: Hosuay and Contributors
   - Description: From README
   - Keywords: cannabis, breeding, machine learning, sacred geometry
   - License: MIT
4. Publish and get DOI

---

## Version History (Sacred Numbers)

| Version | Release Date | Sacred Alignment | Notes |
|---------|--------------|------------------|-------|
| 3.0.0   | 2025-01-XX   | Trinity (3)      | Research suite transformation |
| 6.0.0   | Future       | Hexagon (6)      | Multi-language integration |
| 9.0.0   | Future       | Completion (9)   | Production platform |
| 27.0.0  | Future       | Cube (3³)        | Federated learning |
| 369.0.0 | Future       | Divine (369)     | Ultimate vision |

---

## Sacred Geometry Commit Convention

All release commits follow:

```
[HARMONIC-369] RELEASE: Version X.Y.Z

Description of release with sacred geometry alignment.

🤖 Generated with Claude Code
```

---

## Post-Release Tasks

- [ ] Monitor GitHub issues for bug reports
- [ ] Update project roadmap
- [ ] Begin planning next release
- [ ] Archive old documentation versions
- [ ] Update benchmark comparisons

---

**Current Release**: v3.0.0
**Next Planned Release**: v3.1.0 (Feature additions)
**Sacred Geometry**: Aligned with 369 principles ✨
