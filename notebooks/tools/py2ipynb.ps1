# Converts a percent-format Python script (# %% / # %% [markdown] markers) into a Jupyter notebook.
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Destination
)
$ErrorActionPreference = 'Stop'
$lines = Get-Content -Path $Source -Encoding UTF8
$script:cells = New-Object System.Collections.ArrayList
$script:type = $null
$script:buf = New-Object System.Collections.ArrayList

function Flush {
    if ($null -eq $script:type) { return }
    while ($script:buf.Count -gt 0 -and $script:buf[$script:buf.Count - 1].Trim() -eq '') { $script:buf.RemoveAt($script:buf.Count - 1) }
    while ($script:buf.Count -gt 0 -and $script:buf[0].Trim() -eq '') { $script:buf.RemoveAt(0) }
    if ($script:buf.Count -eq 0) { return }
    $src = @()
    for ($i = 0; $i -lt $script:buf.Count; $i++) {
        $t = [string]$script:buf[$i]
        if ($script:type -eq 'markdown') {
            if ($t.StartsWith('# ')) { $t = $t.Substring(2) } elseif ($t -eq '#') { $t = '' }
        }
        if ($i -lt $script:buf.Count - 1) { $t += "`n" }
        $src += $t
    }
    $cell = [ordered]@{ cell_type = $script:type; metadata = @{}; source = $src }
    if ($script:type -eq 'code') { $cell['execution_count'] = $null; $cell['outputs'] = @() }
    [void]$script:cells.Add($cell)
    $script:buf.Clear()
}

foreach ($line in $lines) {
    if ($line -match '^# %% \[markdown\]') { Flush; $script:type = 'markdown'; continue }
    if ($line -match '^# %%') { Flush; $script:type = 'code'; continue }
    if ($null -ne $script:type) { [void]$script:buf.Add($line) }
}
Flush

$nb = [ordered]@{
    cells          = $script:cells
    metadata       = [ordered]@{
        colab         = @{ provenance = @(); gpuType = 'T4' }
        kernelspec    = @{ display_name = 'Python 3'; name = 'python3' }
        language_info = @{ name = 'python' }
        accelerator   = 'GPU'
    }
    nbformat       = 4
    nbformat_minor = 5
}
$json = $nb | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($Destination, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Wrote $Destination with $($script:cells.Count) cells"

