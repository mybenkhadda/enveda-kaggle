<#
Shared helpers for the Colab Drive migration scripts (dot-source this file).

  Read-AssetMap <path>          strict reader for the RESTRICTED YAML of configs/drive_asset_map.yaml
                                (2-space indentation, key: value scalars, inline [a, b] lists, # comments).
                                Anything else is rejected loudly instead of being guessed.
  Resolve-EnvedaTarget          <DriveRoot> -> <DriveRoot>\EnvedaCASMI (no double nesting if already given)
  Format-Bytes                  human-readable size

Windows PowerShell 5.1 compatible. Nothing here copies, moves or deletes anything.
#>

function Remove-YamlComment([string]$Line) {
    $inSingle = $false; $inDouble = $false
    for ($i = 0; $i -lt $Line.Length; $i++) {
        $c = $Line[$i]
        if ($c -eq "'" -and -not $inDouble) { $inSingle = -not $inSingle; continue }
        if ($c -eq '"' -and -not $inSingle) { $inDouble = -not $inDouble; continue }
        if ($c -eq '#' -and -not $inSingle -and -not $inDouble -and ($i -eq 0 -or $Line[$i - 1] -match '\s')) {
            return $Line.Substring(0, $i)
        }
    }
    return $Line
}

function ConvertFrom-YamlScalar([string]$Value) {
    $v = $Value.Trim()
    if ($v -eq '' -or $v -eq 'null' -or $v -eq '~') { return $null }
    if ($v -eq 'true') { return $true }
    if ($v -eq 'false') { return $false }
    if ($v.StartsWith('[')) {
        if (-not $v.EndsWith(']')) { throw "unterminated inline list: $Value" }
        $inner = $v.Substring(1, $v.Length - 2).Trim()
        $items = New-Object System.Collections.ArrayList
        if ($inner -ne '') {
            foreach ($part in $inner.Split(',')) {
                [void]$items.Add((ConvertFrom-YamlScalar $part))
            }
        }
        return , ([object[]]$items.ToArray())
    }
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
        return $v.Substring(1, $v.Length - 2)
    }
    return $v
}

function Read-AssetMap([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "asset map not found: $Path" }
    $root = [ordered]@{}
    $section = $null
    $asset = $null
    $n = 0
    foreach ($raw in (Get-Content -LiteralPath $Path -Encoding UTF8)) {
        $n++
        $line = (Remove-YamlComment $raw).TrimEnd()
        if ($line.Trim() -eq '') { continue }
        if ($line.Contains("`t")) { throw "line ${n}: tabs are not allowed in $Path" }
        $indent = $line.Length - $line.TrimStart(' ').Length
        $t = $line.Trim()
        if ($t -notmatch '^([A-Za-z0-9_.\-]+):(\s+(.*))?$') { throw "line ${n}: unsupported YAML syntax: $raw" }
        $key = $Matches[1]
        $val = if ($Matches[3]) { $Matches[3] } else { '' }
        if ($indent -eq 0) {
            if ($val -eq '') { $root[$key] = [ordered]@{}; $section = $key; $asset = $null }
            else { $root[$key] = ConvertFrom-YamlScalar $val; $section = $null; $asset = $null }
        }
        elseif ($indent -eq 2) {
            if ($null -eq $section) { throw "line ${n}: indented key without a parent section" }
            if ($val -eq '') { $root[$section][$key] = [ordered]@{}; $asset = $key }
            else { $root[$section][$key] = ConvertFrom-YamlScalar $val; $asset = $null }
        }
        elseif ($indent -eq 4) {
            if ($null -eq $asset) { throw "line ${n}: 4-space key without a parent entry" }
            $root[$section][$asset][$key] = ConvertFrom-YamlScalar $val
        }
        else { throw "line ${n}: unsupported indentation ($indent spaces)" }
    }
    foreach ($req in 'drive_root_name', 'directories', 'assets') {
        if (-not $root.Contains($req)) { throw "asset map is missing '$req'" }
    }
    foreach ($name in $root['assets'].Keys) {
        $a = $root['assets'][$name]
        foreach ($f in 'source', 'destination', 'mode', 'required', 'category') {
            if (-not $a.Contains($f)) { throw "asset '$name' is missing '$f'" }
        }
        if ($a['mode'] -notin 'file', 'directory') { throw "asset '$name': mode must be file|directory" }
        foreach ($p in $a['source'], $a['destination']) {
            if ([System.IO.Path]::IsPathRooted($p) -or $p -match '(^|[\\/])\.\.([\\/]|$)') { throw "asset '$name': paths must be relative without '..': $p" }
        }
    }
    return $root
}

function Resolve-EnvedaTarget([string]$DriveRoot, [string]$Name) {
    $full = [System.IO.Path]::GetFullPath($DriveRoot)
    if ((Split-Path -Leaf $full) -eq $Name) { return $full }
    return (Join-Path $full $Name)
}

function Format-Bytes([double]$Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N1} MB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N1} KB' -f ($Bytes / 1KB)) }
    return ('{0} B' -f [int64]$Bytes)
}

function ConvertTo-WinRel([string]$Rel) { return $Rel.Replace('/', '\') }
