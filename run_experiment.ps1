param(
    [int]$Rounds = 50
)

$source = (Get-Location).Path
$destination = "C:\work\Graduation_Research"
$python = "C:\Users\hideb\.venvs\Graduation_Research\Scripts\python.exe"
$flwr = "C:\Users\hideb\.venvs\Graduation_Research\Scripts\flwr.exe"

New-Item -ItemType Directory -Force $destination | Out-Null
Get-ChildItem -Force $source |
    Where-Object { $_.Name -notin @(".git", ".venv", "logs", "fixed_eval", "__pycache__", ".pytest_cache", "build", "dist") } |
    ForEach-Object { Copy-Item $_.FullName -Destination $destination -Recurse -Force }

Remove-Item "$destination\logs", "$destination\fixed_eval" -Recurse -Force -ErrorAction SilentlyContinue

& $flwr federation simulation-config @none/default local --num-supernodes 10 --init-args-num-cpus 10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location $destination
try {
    & $flwr run . --stream --run-config "num-server-rounds=$Rounds"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}