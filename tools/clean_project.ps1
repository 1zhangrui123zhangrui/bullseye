param([switch]$KeepOutputs)
$root = Split-Path -Parent $PSScriptRoot
Get-ChildItem $root -Directory -Filter '__pycache__' -Recurse -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
if (-not $KeepOutputs) {
    Get-ChildItem $root -Directory -Filter 'output*' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem (Join-Path $root 'models') -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq '.pt' -or $_.Name -like '*_labels.txt' } |
    Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host 'Cleaned MaixVision project cache and non-runtime artifacts.'
