<#
.SYNOPSIS
    Build the PRIVATE Kaggle Model payload (enveda-casmi-v2-a7 / sklearn / full-e2e) locally. Uploads NOTHING.

.DESCRIPTION
    Creates
        <OutputRoot>\metadata\model-metadata.json            (for: kaggle models create -p <OutputRoot>\metadata)
        <OutputRoot>\metadata\model-instance-metadata.json   (record / CLI alternative; KaggleHub does not need it)
        <OutputRoot>\payload\PACKAGE_INFO.json
        <OutputRoot>\payload\kaggle_model_inventory.json
        <OutputRoot>\payload\bundle\...                      the WHOLE frozen bundle tree (hard links when possible)
        <OutputRoot>\payload\runtime\src\casmi_runtime\...   orchestration only (explicit file list, incl. named_inference.py)
        <OutputRoot>\payload\runtime\notebooks\02_kaggle_inference.ipynb, requirements-kaggle.txt, KAGGLE_RUN_GUIDE.md

    Bundle files are HARD-LINKED (same NTFS volume: no second copy of the ~GB reference arrays); if a link cannot be
    made (other volume, OneDrive online-only placeholder, ...) the file is copied. NEVER edit files under
    payload\bundle: a hard link IS the original file.
    Excluded everywhere: __pycache__, *.pyc, .ipynb_checkpoints, .git. Nothing else is filtered out of the bundle.
    The payload never contains runtime\src\casmi_infer (the frozen bundle\code\casmi_infer is the only inference code)
    nor any competition file (test.parquet / sample_submission.csv / train.parquet).
    Ends with a STATIC filename check against kaggle_model_inventory.json + the bundle manifest (no hashing) and prints
    KAGGLE MODEL PAYLOAD READY.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\prepare_kaggle_model.ps1 -KaggleUsername YOUR_KAGGLE_USERNAME -Force
#>
[CmdletBinding()]
param(
    [string]$RepoRoot,
    [string]$BundleRoot,
    [string]$OutputRoot,
    [string]$KaggleUsername,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$Placeholder = 'YOUR_KAGGLE_USERNAME'
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-Utf8([string]$Path, [string]$Text) { [System.IO.File]::WriteAllText($Path, $Text, $Utf8NoBom) }
function Read-Json([string]$Path) { return (Get-Content -Raw -LiteralPath $Path -Encoding UTF8 | ConvertFrom-Json) }
function Win([string]$Rel) { return $Rel.Replace('/', '\') }
function Is-Excluded([string]$Rel) {
    $parts = $Rel.Split('\')
    foreach ($p in $parts) { if ($p -in @('__pycache__', '.ipynb_checkpoints', '.git')) { return $true } }
    return $Rel.ToLower().EndsWith('.pyc')
}

# ---- arguments ---------------------------------------------------------------------------------------------------
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
if (-not $BundleRoot) { $BundleRoot = Join-Path $RepoRoot 'bundle' }
$BundleRoot = [System.IO.Path]::GetFullPath($BundleRoot)
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'kaggle_model_build' }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $KaggleUsername -or $KaggleUsername -eq $Placeholder) { throw "pass -KaggleUsername <your Kaggle username> (the metadata owner of the private model)" }
if ($KaggleUsername -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') { throw "-KaggleUsername '$KaggleUsername' does not look like a Kaggle username" }

$InventoryPath = Join-Path $RepoRoot 'kaggle_model_inventory.json'
$TemplateDir = Join-Path $RepoRoot 'kaggle_model'
foreach ($p in @($InventoryPath, (Join-Path $TemplateDir 'model-metadata.template.json'), (Join-Path $TemplateDir 'model-instance-metadata.template.json'),
                 (Join-Path $RepoRoot 'src\casmi_runtime\named_inference.py'), (Join-Path $BundleRoot 'config.json'),
                 (Join-Path $BundleRoot 'manifest.json'), (Join-Path $BundleRoot 'code\casmi_infer\__init__.py'))) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "required input missing: $p" }
}
$Inv = Read-Json $InventoryPath
$Config = Read-Json (Join-Path $BundleRoot 'config.json')
$Manifest = Read-Json (Join-Path $BundleRoot 'manifest.json')
$ModelInfo = Read-Json (Join-Path $BundleRoot 'models\model_info.json')
$ConfigText = Get-Content -Raw -LiteralPath (Join-Path $BundleRoot 'config.json') -Encoding UTF8

# frozen identity of the SOURCE bundle (filename-only mode: identity from JSON, no hashing)
$idProblems = @()
if ($Config.bundle_version -ne $Inv.model.bundle_version) { $idProblems += "config bundle_version $($Config.bundle_version) != $($Inv.model.bundle_version)" }
if ($Config.CONFIG_HASH -ne $Inv.model.CONFIG_HASH) { $idProblems += "config CONFIG_HASH $($Config.CONFIG_HASH) != $($Inv.model.CONFIG_HASH)" }
if ($Config.model_id -ne $Inv.model.model_id) { $idProblems += "config model_id $($Config.model_id) != $($Inv.model.model_id)" }
if ($ModelInfo.model_id -ne $Inv.model.model_id) { $idProblems += "model_info model_id $($ModelInfo.model_id) != $($Inv.model.model_id)" }
if ($ModelInfo.freeze_status -ne 'FROZEN') { $idProblems += "model_info freeze_status $($ModelInfo.freeze_status) != FROZEN" }
if ($Config.aggregator_name -ne $Inv.model.aggregator) { $idProblems += "config aggregator_name $($Config.aggregator_name) != $($Inv.model.aggregator)" }
$TempRaw = [regex]::Match($ConfigText, '"calibration_temperature"\s*:\s*([0-9eE+.\-]+)').Groups[1].Value
if ($TempRaw -ne '1.1947045372735254') { $idProblems += "config calibration_temperature '$TempRaw' != 1.1947045372735254" }
if ($idProblems.Count) { throw ("source bundle is not the frozen v2-A7 identity:`n  " + ($idProblems -join "`n  ")) }

# ---- output folder -----------------------------------------------------------------------------------------------
foreach ($guarded in @($RepoRoot, $BundleRoot)) {
    if ($OutputRoot.TrimEnd('\') -ieq $guarded.TrimEnd('\') -or $guarded.StartsWith($OutputRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "-OutputRoot $OutputRoot would contain / replace $guarded"
    }
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (-not $Force) { throw "$OutputRoot already exists. Re-run with -Force to rebuild it." }
    $unexpected = Get-ChildItem -LiteralPath $OutputRoot -Force | Where-Object { $_.Name -notin @('payload', 'metadata') }
    if ($unexpected) { throw "$OutputRoot holds unexpected entries ($($unexpected.Name -join ', ')) -- refusing to delete it" }
    # deleting a hard link removes only that directory entry; the original bundle files are untouched
    Remove-Item -LiteralPath $OutputRoot -Recurse -Force
}
$Payload = Join-Path $OutputRoot 'payload'
$MetaDir = Join-Path $OutputRoot 'metadata'
New-Item -ItemType Directory -Force -Path $Payload, $MetaDir | Out-Null

# ---- 1. the WHOLE frozen bundle (hard links, copy fallback) ------------------------------------------------------
$nLinked = 0; $nCopied = 0; $bytesCopied = 0; $nSkipped = 0
$bundleFiles = Get-ChildItem -LiteralPath $BundleRoot -Recurse -File -Force
foreach ($f in $bundleFiles) {
    $rel = $f.FullName.Substring($BundleRoot.TrimEnd('\').Length + 1)
    if (Is-Excluded $rel) { $nSkipped++; continue }
    $dst = Join-Path (Join-Path $Payload 'bundle') $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    try {
        New-Item -ItemType HardLink -Path $dst -Target $f.FullName -ErrorAction Stop | Out-Null
        $nLinked++
    } catch {
        Copy-Item -LiteralPath $f.FullName -Destination $dst -Force
        $nCopied++; $bytesCopied += $f.Length
    }
}
Write-Host ("bundle staged: {0} hard-linked, {1} copied ({2:N2} GB copied), {3} cache files skipped" -f $nLinked, $nCopied, ($bytesCopied / 1GB), $nSkipped)

# ---- 2. runtime orchestration (explicit list from the inventory; small files are copied) ---------------------------
$nRuntime = 0
foreach ($prop in $Inv.runtime_source.training_repo_sources.PSObject.Properties) {
    $src = Join-Path $RepoRoot (Win $prop.Value)
    $dst = Join-Path $Payload (Win $prop.Name)
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { throw "missing runtime source file: $src" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
    $nRuntime++
}
Copy-Item -LiteralPath $InventoryPath -Destination (Join-Path $Payload 'kaggle_model_inventory.json') -Force

# ---- 3. PACKAGE_INFO.json ---------------------------------------------------------------------------------------------
$Handle = "$KaggleUsername/$($Inv.model.slug)/$($Inv.model.framework)/$($Inv.model.variation)"
$info = [ordered]@{
    package_format = 'casmi-kaggle-model-payload-1'
    model_slug = $Inv.model.slug; framework = $Inv.model.framework; variation = $Inv.model.variation; private = $true
    handle = $Handle
    bundle_version = $Config.bundle_version; manifest_bundle_version = $Manifest.bundle_version
    CONFIG_HASH = $Config.CONFIG_HASH; model_id = $Config.model_id; aggregator = $Config.aggregator_name
    calibration_temperature = '__CALIBRATION_TEMPERATURE__'
    model_freeze_status = $ModelInfo.freeze_status
    bundle_integrity_mode = 'FILENAMES_ONLY'; bundle_hash_verification = $false; fixture_hash_verification = $false
    entry_notebook = $Inv.model.entry.notebook; entry_function = $Inv.model.entry.function
    scientific_authority = 'bundle/code/casmi_infer'
    competition_files_included = $false; competition_required_separately = @('test.parquet', 'sample_submission.csv')
    n_bundle_files = ($nLinked + $nCopied); n_runtime_files = $nRuntime
    built_at_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    built_from = [ordered]@{ repo_root = $RepoRoot; bundle_root = $BundleRoot }
}
$infoJson = ($info | ConvertTo-Json -Depth 8).Replace('"__CALIBRATION_TEMPERATURE__"', $TempRaw)
Write-Utf8 (Join-Path $Payload 'PACKAGE_INFO.json') $infoJson

# ---- 4. metadata (username materialized; nothing is uploaded) ------------------------------------------------------
foreach ($name in @('model-metadata', 'model-instance-metadata')) {
    $t = Get-Content -Raw -LiteralPath (Join-Path $TemplateDir "$name.template.json") -Encoding UTF8
    Write-Utf8 (Join-Path $MetaDir "$name.json") $t.Replace($Placeholder, $KaggleUsername)
}

# ---- 5. STATIC filename check (no hashing) ----------------------------------------------------------------------------
$problems = @()
$required = @($Inv.payload_top_level.required) + @($Inv.runtime_source.required) + @($Inv.frozen_bundle.required)
foreach ($r in $required) { if (-not (Test-Path -LiteralPath (Join-Path $Payload (Win $r)) -PathType Leaf)) { $problems += "missing file: $r" } }
foreach ($d in $Inv.frozen_bundle.required_directories) { if (-not (Test-Path -LiteralPath (Join-Path $Payload (Win $d)) -PathType Container)) { $problems += "missing directory: $d" } }
foreach ($m in $Manifest.files.PSObject.Properties.Name) { if (-not (Test-Path -LiteralPath (Join-Path $Payload ('bundle\' + (Win $m))) -PathType Leaf)) { $problems += "manifest file not staged: bundle/$m" } }
foreach ($fb in $Inv.runtime_source.forbidden) { if (Test-Path -LiteralPath (Join-Path $Payload (Win $fb))) { $problems += "forbidden in payload: $fb" } }
$staged = Get-ChildItem -LiteralPath $Payload -Recurse -File -Force
foreach ($f in $staged) {
    $rel = $f.FullName.Substring($Payload.Length + 1)
    if ($f.Name -in @($Inv.competition.forbidden_in_payload)) { $problems += "competition file in payload: $rel" }
    if (Is-Excluded $rel) { $problems += "cache file in payload: $rel" }
    if ($f.Length -eq 0 -and $f.Name -ne '__init__.py') { $problems += "empty file: $rel" }   # empty package inits are legitimate
}
$pi = Read-Json (Join-Path $Payload 'PACKAGE_INFO.json')
if ($pi.bundle_version -ne 'v2-A7' -or $pi.model_id -ne 'V1_TL_1K_TESTSIM_STRICT' -or $pi.CONFIG_HASH -ne '60174e39a2a3c6b4') { $problems += 'PACKAGE_INFO.json identity mismatch' }
if ($problems.Count) {
    $problems | ForEach-Object { Write-Host "PROBLEM: $_" }
    throw "KAGGLE MODEL PAYLOAD NOT READY ($($problems.Count) problems)"
}

$totalBytes = ($staged | Measure-Object -Property Length -Sum).Sum
Write-Host ''
Write-Host 'KAGGLE MODEL PAYLOAD READY'
Write-Host ("  payload path : {0}" -f $Payload)
Write-Host ("  total files  : {0}" -f $staged.Count)
Write-Host ("  approx size  : {0:N2} GB" -f ($totalBytes / 1GB))
Write-Host ("  metadata     : {0}" -f $MetaDir)
Write-Host ("  handle       : {0}" -f $Handle)
Write-Host ''
Write-Host 'NEXT (run them yourself; this script uploaded nothing):'
Write-Host "  kaggle models create -p `"$MetaDir`"          # once: creates the PRIVATE parent model"
Write-Host "  python .\scripts\upload_kaggle_model.py --handle $Handle"
