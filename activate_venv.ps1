$venvPath = "C:\Users\shrut\sonic_dashboard\.venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    & $venvPath
}
