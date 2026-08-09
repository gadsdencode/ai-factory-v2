[CmdletBinding()]
param(
    [ValidateSet("8gb-safe", "12gb-safe")]
    [string]$Profile = "12gb-safe",

    [switch]$RunContainerCheck
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$minimumWslVersion = [version]"2.1.5"
$minimumFreeDiskGiB = 60
$minimumVramMiB = @{
    "8gb-safe" = 7680
    "12gb-safe" = 11776
}
$minimumRamGiB = @{
    "8gb-safe" = 24
    "12gb-safe" = 48
}
$failures = [System.Collections.Generic.List[string]]::new()
$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Add-CheckFailure {
    param(
        [Parameter(Mandatory)]
        [string]$Message
    )

    $failures.Add($Message)
    Write-Host "  FAIL: $Message" -ForegroundColor Red
}

function Invoke-DisplayedCommand {
    param(
        [Parameter(Mandatory)]
        [string]$Label,

        [Parameter(Mandatory)]
        [scriptblock]$Command
    )

    Write-Host "`n$Label"
    $global:LASTEXITCODE = 0
    try {
        & $Command
        if ($LASTEXITCODE -ne 0) {
            Add-CheckFailure "$Label exited with code $LASTEXITCODE."
        }
    }
    catch {
        Add-CheckFailure "$Label failed: $($_.Exception.Message)"
    }
}

Write-Host "AI Factory Docker Desktop host check"
Write-Host "Profile: $Profile"

Push-Location $repositoryRoot
try {
    Write-Host "`nSystem memory"
    try {
        $computer = Get-CimInstance Win32_ComputerSystem
        $ramGiB = [math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
        Write-Host "  Installed RAM: $ramGiB GiB"
        if ($ramGiB -lt $minimumRamGiB[$Profile]) {
            Add-CheckFailure (
                "Profile '$Profile' expects at least " +
                "$($minimumRamGiB[$Profile]) GiB RAM."
            )
        }
    }
    catch {
        Write-Warning "Could not query installed RAM: $($_.Exception.Message)"
    }

    Write-Host "`nDefault Docker data drive"
    try {
        $localAppData = Get-Item $env:LOCALAPPDATA
        $dataDrive = Get-PSDrive -Name $localAppData.PSDrive.Name
        $freeDiskGiB = [math]::Round($dataDrive.Free / 1GB, 1)
        Write-Host "  Drive: $($dataDrive.Name):"
        Write-Host "  Free space: $freeDiskGiB GiB"
        if ($freeDiskGiB -lt $minimumFreeDiskGiB) {
            Write-Warning (
                "The default Docker data drive has less than " +
                "$minimumFreeDiskGiB GiB free. Free space or confirm that " +
                "Docker Desktop stores its disk image on another drive."
            )
        }
    }
    catch {
        Write-Warning (
            "Could not query the default Docker data drive. If Docker Desktop " +
            "stores its disk image elsewhere, verify at least " +
            "$minimumFreeDiskGiB GiB is free there."
        )
    }

    Write-Host "`nWSL"
    try {
        $global:LASTEXITCODE = 0
        $wslVersionOutput = & wsl.exe --version 2>&1
        $wslVersionOutput | ForEach-Object { Write-Host "  $_" }
        if ($LASTEXITCODE -ne 0) {
            Add-CheckFailure "wsl.exe --version exited with code $LASTEXITCODE."
        }
        else {
            $versionMatch = [regex]::Match(
                ($wslVersionOutput -join "`n"),
                "WSL version:\s*([0-9.]+)",
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
            )
            if (-not $versionMatch.Success) {
                Add-CheckFailure "Could not parse the installed WSL version."
            }
            else {
                $installedWslVersion = [version]$versionMatch.Groups[1].Value
                if ($installedWslVersion -lt $minimumWslVersion) {
                    Add-CheckFailure (
                        "WSL $installedWslVersion is below the Docker Desktop " +
                        "minimum $minimumWslVersion. Run 'wsl --update'."
                    )
                }
            }
        }
    }
    catch {
        Add-CheckFailure "WSL is unavailable: $($_.Exception.Message)"
    }

    Invoke-DisplayedCommand "WSL distributions" { wsl.exe --list --verbose }

    Write-Host "`nNVIDIA GPU"
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        Add-CheckFailure "nvidia-smi is not available on PATH."
    }
    else {
        $global:LASTEXITCODE = 0
        $gpuRows = @(
            & $nvidiaSmi.Source `
                --query-gpu=name,memory.total,driver_version `
                --format=csv,noheader,nounits 2>&1
        )
        $gpuRows | ForEach-Object { Write-Host "  $_" }
        if ($LASTEXITCODE -ne 0) {
            Add-CheckFailure "nvidia-smi exited with code $LASTEXITCODE."
        }
        elseif ($gpuRows.Count -eq 0) {
            Add-CheckFailure "nvidia-smi did not return a GPU."
        }
        else {
            $gpuFields = ([string]$gpuRows[0]) -split ",\s*"
            if ($gpuFields.Count -lt 3) {
                Add-CheckFailure "Could not parse nvidia-smi GPU output."
            }
            else {
                $gpuMemoryMiB = 0
                if (-not [int]::TryParse($gpuFields[1], [ref]$gpuMemoryMiB)) {
                    Add-CheckFailure "Could not parse GPU memory from nvidia-smi."
                }
                elseif ($gpuMemoryMiB -lt $minimumVramMiB[$Profile]) {
                    Add-CheckFailure (
                        "Profile '$Profile' requires at least " +
                        "$($minimumVramMiB[$Profile]) MiB total VRAM; " +
                        "device 0 reports $gpuMemoryMiB MiB."
                    )
                }
            }
        }
    }

    Invoke-DisplayedCommand "Docker client and server" { docker version }
    Invoke-DisplayedCommand "Docker Compose" { docker compose version }

    Write-Host "`nDocker container mode"
    try {
        $global:LASTEXITCODE = 0
        $dockerOsType = (& docker info --format "{{.OSType}}" 2>&1).Trim()
        Write-Host "  OSType: $dockerOsType"
        if ($LASTEXITCODE -ne 0) {
            Add-CheckFailure "docker info exited with code $LASTEXITCODE."
        }
        elseif ($dockerOsType -ne "linux") {
            Add-CheckFailure "Docker Desktop must be using Linux containers."
        }
    }
    catch {
        Add-CheckFailure "Could not query Docker container mode: $($_.Exception.Message)"
    }

    if ($RunContainerCheck) {
        Invoke-DisplayedCommand "Container GPU and storage preflight" {
            docker compose run --rm ai-factory `
                python docker/verify_runtime.py `
                --require-gpu `
                --profile $Profile
        }
    }
    else {
        Write-Host "`nContainer check skipped. After building, run:"
        Write-Host (
            ".\docker\Test-DockerDesktopHost.ps1 -Profile $Profile " +
            "-RunContainerCheck"
        )
    }
}
finally {
    Pop-Location
}

if ($failures.Count -gt 0) {
    Write-Host "`nHost verification failed with $($failures.Count) issue(s)." `
        -ForegroundColor Red
    exit 1
}

Write-Host "`nHost verification passed for profile '$Profile'." `
    -ForegroundColor Green
