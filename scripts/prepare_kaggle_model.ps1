param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$BundleRoot = "",
    [string]$OutputRoot = "",
    [string]$KaggleUsername = "YOUR_KAGGLE_USERNAME",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path $RepoRoot).Path
if (-not $BundleRoot) { $BundleRoot = Join-Path $RepoRoot "bundle" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "kaggle_model_build" }

$BundleRoot = (Resolve-Path $BundleRoot).Path
$Payload = Join-Path $OutputRoot "payload"
$MetaRoot = Join-Path $OutputRoot "metadata"

if (Test-Path $OutputRoot) {
    if (-not $Force) {
        throw "$OutputRoot already exists. Re-run with -Force to rebuild it."
    }
    Remove-Item -Recurse -Force $OutputRoot
}

New-Item -ItemType Directory -Force -Path $Payload, $MetaRoot | Out-Null

function Copy-OrHardLinkFile {
    param([Parameter(Mandatory=$true)][string]$Source,
          [Parameter(Mandatory=$true)][string]$Destination)

    $parent = Split-Path -Parent $Destination
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    try {
        New-Item -ItemType HardLink -Path $Destination -Target $Source -ErrorAction Stop | Out-Null
    }
    catch {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Copy-TreeLinked {
    param([Parameter(Mandatory=$true)][string]$SourceRoot,
          [Parameter(Mandatory=$true)][string]$DestinationRoot)

    if (-not (Test-Path $SourceRoot)) { throw "Missing source tree: $SourceRoot" }

    Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | Where-Object {
        $_.FullName -notmatch "\\__pycache__\\" -and
        $_.FullName -notmatch "\\.ipynb_checkpoints\\"
    } | ForEach-Object {
        $rel = $_.FullName.Substring($SourceRoot.Length).TrimStart("\", "/")
        Copy-OrHardLinkFile -Source $_.FullName -Destination (Join-Path $DestinationRoot $rel)
    }
}

Write-Host "Preparing Kaggle Model payload..."
Write-Host "Repo   : $RepoRoot"
Write-Host "Bundle : $BundleRoot"
Write-Host "Output : $OutputRoot"

# 1. Entire frozen bundle. This is the scientific authority.
Copy-TreeLinked -SourceRoot $BundleRoot -DestinationRoot (Join-Path $Payload "bundle")

# 2. Lightweight orchestration only. Do NOT copy repo src/casmi_infer: bundle/code is authoritative.
Copy-TreeLinked -SourceRoot (Join-Path $RepoRoot "src\casmi_runtime") -DestinationRoot (Join-Path $Payload "runtime\src\casmi_runtime")

$runtimeFiles = @(
    "notebooks\02_kaggle_inference.ipynb",
    "requirements-kaggle.txt",
    "KAGGLE_RUN_GUIDE.md",
    "kaggle_model_inventory.json"
)
foreach ($rel in $runtimeFiles) {
    $src = Join-Path $RepoRoot $rel
    if (-not (Test-Path $src)) { throw "Missing runtime packaging file: $src" }
    $dst = if ($rel -eq "kaggle_model_inventory.json") {
        Join-Path $Payload "kaggle_model_inventory.json"
    } else {
        Join-Path $Payload ("runtime\" + $rel)
    }
    $dstParent = Split-Path -Parent $dst
    if (-not (Test-Path $dstParent)) {
        New-Item -ItemType Directory -Force -Path $dstParent | Out-Null
    }
    Copy-Item -LiteralPath $src -Destination $dst -Force
}

# 3. Materialize model metadata from templates.
$modelTemplate = Get-Content (Join-Path $RepoRoot "kaggle_model\model-metadata.template.json") -Raw
$instanceTemplate = Get-Content (Join-Path $RepoRoot "kaggle_model\model-instance-metadata.template.json") -Raw
$modelTemplate = $modelTemplate.Replace("YOUR_KAGGLE_USERNAME", $KaggleUsername)
$instanceTemplate = $instanceTemplate.Replace("YOUR_KAGGLE_USERNAME", $KaggleUsername)
$modelTemplate | Set-Content -Encoding UTF8 (Join-Path $MetaRoot "model-metadata.json")
$instanceTemplate | Set-Content -Encoding UTF8 (Join-Path $MetaRoot "model-instance-metadata.json")

# 4. Static inventory validation. No hashes.
$inventory = Get-Content (Join-Path $RepoRoot "kaggle_model_inventory.json") -Raw | ConvertFrom-Json
$required = @()
$required += $inventory.runtime_source.required
$required += $inventory.frozen_bundle.required

$missing = @()
foreach ($rel in $required) {
    $p = Join-Path $Payload ([string]$rel)
    if (-not (Test-Path $p)) { $missing += $rel }
}
foreach ($rel in $inventory.frozen_bundle.required_directories) {
    $p = Join-Path $Payload ([string]$rel)
    if (-not (Test-Path $p -PathType Container)) { $missing += $rel }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing required payload entries:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "Kaggle Model payload is incomplete."
}

# 5. Human-readable package marker.
$marker = @{
    package = "enveda-casmi-v2-a7"
    variation = "full-e2e"
    framework = "sklearn"
    input_mode = "FILENAMES_ONLY"
    bundle_version = "v2-A7"
    model_id = "V1_TL_1K_TESTSIM_STRICT"
    aggregator = "MOST_CONFIDENT_SPECTRUM"
    calibration_temperature = 1.1947045372735254
    runtime_entry = "runtime/src/casmi_runtime/named_inference.py"
    frozen_code = "bundle/code/casmi_infer"
    competition_is_separate = $true
} | ConvertTo-Json -Depth 5
$marker | Set-Content -Encoding UTF8 (Join-Path $Payload "PACKAGE_INFO.json")

$totalBytes = (Get-ChildItem -LiteralPath $Payload -Recurse -File | Measure-Object -Property Length -Sum).Sum
$fileCount = (Get-ChildItem -LiteralPath $Payload -Recurse -File).Count

Write-Host ""
Write-Host "KAGGLE MODEL PAYLOAD READY" -ForegroundColor Green
Write-Host "Payload : $Payload"
Write-Host "Files   : $fileCount"
Write-Host ("Size    : {0:N2} GB" -f ($totalBytes / 1GB))
Write-Host "Metadata: $MetaRoot"
Write-Host ""
Write-Host "Competition files remain separate: test.parquet + sample_submission.csv"
Write-Host "Nothing was uploaded to Kaggle by this preparation script."
