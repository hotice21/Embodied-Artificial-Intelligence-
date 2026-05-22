param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

& (Join-Path $PSScriptRoot "run_in_env.ps1") serpent.exe @Args
exit $LASTEXITCODE
