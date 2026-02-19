param(
    [string]$TargetPath = "C:\Users\hellc\Google Drive\TFG\TFG_AOS0020---TEST"
)

# Remove desktop.ini files under TargetPath excluding .git folder
Get-ChildItem -Path $TargetPath -Filter 'desktop.ini' -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\\.git\\' } |
    ForEach-Object {
        try {
            Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
        } catch {
            # ignore errors
        }
    }

# Also remove any leftover hidden/system desktop.ini in root
$root = Resolve-Path -Path $TargetPath
$rootDesktop = Join-Path -Path $root -ChildPath 'desktop.ini'
if (Test-Path $rootDesktop) { Remove-Item -LiteralPath $rootDesktop -Force -ErrorAction SilentlyContinue }
