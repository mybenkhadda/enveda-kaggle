<#
.SYNOPSIS
    Build the lightweight GitHub RUNTIME repo (code only) for Colab / Kaggle from this training repo.

.DESCRIPTION
    Copies exactly what runtime\runtime_files.json lists:
      * the whitelisted src\casmi modules (runtime-test import closure + the modules the bundle ships; nothing training-only)
      * src\casmi_infer\, src\casmi_runtime\ (incl. named_inference.py, the Kaggle E2E entry)   (caches excluded)
      * scripts (v63_bundle_check, check_runtime_repo, detect_accelerator), kaggle\kaggle_submit.ipynb, tests,
        notebooks 00-03, requirements-colab/kaggle, README, COLAB/KAGGLE run guides, .gitignore, runtime_files.json,
        kaggle_model_inventory.json   (the Kaggle MODEL payload is built separately by scripts\prepare_kaggle_model.ps1)
    This folder is the GitHub runtime repo. Kaggle does not use it directly: Kaggle runs from the private Kaggle Model
    built by scripts\prepare_kaggle_model.ps1 (bundle\ + runtime\ with the casmi_runtime subset).
    NEVER copied: bundle\, bundle_archive\, data\, outputs\, QCR caches, MOL_DEV / HOST artifacts, training parquet,
    .npy / .parquet files, the 11_00 / 11_01 / 11_02 development notebooks.
    This script never runs git / gh and uploads nothing; it prints the next commands.

.PARAMETER OutDir
    Target folder (default <repo>\colab_repo). A folder outside OneDrive (e.g. C:\dev\enveda-casmi-runtime) keeps
    OneDrive from syncing the .git folder.

.PARAMETER Force
    Rebuild an existing target (everything except its .git folder is replaced). Without -Force nothing is overwritten.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\prepare_runtime_repo.ps1
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$SpecPath = Join-Path $Root 'runtime\runtime_files.json'
if (-not (Test-Path -LiteralPath $SpecPath)) { throw "runtime\runtime_files.json not found under $Root -- run from the training repo" }
$Spec = Get-Content -Raw -LiteralPath $SpecPath | ConvertFrom-Json
if (-not $OutDir) { $OutDir = Join-Path $Root 'colab_repo' }
$OutDir = [System.IO.Path]::GetFullPath($OutDir)

if (Test-Path -LiteralPath $OutDir) {
    if (-not $Force) { throw "$OutDir already exists. Re-run with -Force to rebuild it (its .git folder is kept)." }
    Get-ChildItem -LiteralPath $OutDir -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Copy-Tree([string]$Src, [string]$Dst) {
    robocopy $Src $Dst /E /XD __pycache__ .ipynb_checkpoints .pytest_cache /XF *.pyc *.pyo *.parquet *.npy *.npz *.whl /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $Src -> $Dst" }
}

function Copy-One([string]$Src, [string]$Dst) {
    if (-not (Test-Path -LiteralPath $Src)) { throw "missing source file: $Src" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dst) | Out-Null
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
}

function Win([string]$Rel) { return $Rel.Replace('/', '\') }

# ---- whitelisted casmi modules (one by one: no training code rides along) --------------------------------------
foreach ($m in $Spec.casmi_modules) {
    Copy-One (Join-Path $Root (Win "src/$m")) (Join-Path $OutDir (Win "src/$m"))
}
# ---- inference + orchestration packages ---------------------------------------------------------------------
foreach ($t in $Spec.copy_trees) {
    Copy-Tree (Join-Path $Root (Win $t.src)) (Join-Path $OutDir (Win $t.dst))
}
# ---- scripts, notebooks, tests, docs, requirements, .gitignore, the spec itself ---------------------------------
foreach ($f in $Spec.copy_files) {
    Copy-One (Join-Path $Root (Win $f.src)) (Join-Path $OutDir (Win $f.dst))
}

$files = Get-ChildItem -LiteralPath $OutDir -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\\.git\\' }
$totalMb = [math]::Round((($files | Measure-Object -Property Length -Sum).Sum) / 1MB, 2)
Write-Host ("Runtime repo prepared: {0}  ({1} files, {2} MB)" -f $OutDir, $files.Count, $totalMb)

# ---- next commands: PRINTED ONLY, never executed here ---------------------------------------------------------
$next = @'

NEXT (run them yourself in PowerShell):

  # 1. safety check -- must print RUNTIME REPO CHECK: PASS
  python "<ROOT>\scripts\check_runtime_repo.py" --repo "<OUTDIR>"

  # 2. create the private GitHub repo and push
  cd "<OUTDIR>"
  git init -b main
  git add .
  git commit -m "Initial CASMI GPU runtime"
  gh auth status
  gh repo create <REPO> `
      --private `
      --source . `
      --remote origin `
      --push
  gh repo view --web

  # later updates (after re-running this script with -Force):
  #   cd "<OUTDIR>"; git add -A; git commit -m "Update CASMI runtime"; git push
'@
Write-Host ($next.Replace('<OUTDIR>', $OutDir).Replace('<ROOT>', $Root).Replace('<REPO>', $Spec.repo_name))
