# Penn Treebank Project Cleanup Script for PowerShell
# This script provides easy access to the Python cleanup script

param(
    [switch]$DryRun,
    [switch]$Aggressive,
    [switch]$KeepModels,
    [switch]$Interactive,
    [switch]$Help
)

function Show-Help {
    Write-Host ""
    Write-Host "Penn Treebank Project Cleanup for PowerShell" -ForegroundColor Cyan
    Write-Host "=============================================="
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\cleanup.ps1                    # Interactive menu"
    Write-Host "  .\cleanup.ps1 -DryRun           # Preview cleanup"
    Write-Host "  .\cleanup.ps1 -Aggressive       # Remove everything including models"
    Write-Host "  .\cleanup.ps1 -KeepModels       # Standard cleanup, keep models"
    Write-Host "  .\cleanup.ps1 -Interactive      # Ask before each deletion"
    Write-Host "  .\cleanup.ps1 -Help             # Show this help"
    Write-Host ""
    Write-Host "Benefits:"
    Write-Host "  - Removes extracted Penn Treebank data (~150MB)" -ForegroundColor Green
    Write-Host "  - Clears Python cache and temporary files" -ForegroundColor Green
    Write-Host "  - Removes TensorBoard logs" -ForegroundColor Green
    Write-Host "  - Faster Git pushes and Google Drive uploads" -ForegroundColor Green
    Write-Host ""
}

function Test-ProjectDirectory {
    if (-not (Test-Path "src") -or -not (Test-Path "config")) {
        Write-Host "Error: Please run this script from the project root directory" -ForegroundColor Red
        Write-Host "(Directory should contain 'src' and 'config' folders)" -ForegroundColor Red
        return $false
    }
    return $true
}

function Show-Menu {
    Write-Host ""
    Write-Host "Penn Treebank Project Cleanup" -ForegroundColor Cyan
    Write-Host "=============================="
    Write-Host ""
    Write-Host "Choose cleanup option:"
    Write-Host ""
    Write-Host "1. Preview cleanup (dry-run)" -ForegroundColor Yellow
    Write-Host "2. Standard cleanup (keeps models)" -ForegroundColor Green
    Write-Host "3. Aggressive cleanup (removes models too)" -ForegroundColor Red
    Write-Host "4. Interactive cleanup" -ForegroundColor Blue
    Write-Host "5. Exit"
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-5)"
    return $choice
}

# Main script logic
if ($Help) {
    Show-Help
    exit 0
}

if (-not (Test-ProjectDirectory)) {
    exit 1
}

# If specific parameters are provided, run directly
if ($DryRun -or $Aggressive -or $KeepModels -or $Interactive) {
    $args = @()
    if ($DryRun) { $args += "--dry-run" }
    if ($Aggressive) { $args += "--aggressive" }
    if ($KeepModels) { $args += "--keep-models" }
    if ($Interactive) { $args += "--interactive" }
    
    Write-Host "Running cleanup with options: $($args -join ' ')" -ForegroundColor Cyan
    & python scripts/cleanup_project.py @args
    
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 0
}

# Interactive menu
do {
    $choice = Show-Menu
    
    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "Running preview mode..." -ForegroundColor Yellow
            & python scripts/cleanup_project.py --dry-run
        }
        "2" {
            Write-Host ""
            Write-Host "Running standard cleanup..." -ForegroundColor Green
            & python scripts/cleanup_project.py
        }
        "3" {
            Write-Host ""
            Write-Host "Running aggressive cleanup..." -ForegroundColor Red
            Write-Host "Warning: This will remove model checkpoints!" -ForegroundColor Yellow
            $confirm = Read-Host "Are you sure? (y/N)"
            if ($confirm -eq "y" -or $confirm -eq "Y") {
                & python scripts/cleanup_project.py --aggressive
            } else {
                Write-Host "Cancelled." -ForegroundColor Yellow
            }
        }
        "4" {
            Write-Host ""
            Write-Host "Running interactive cleanup..." -ForegroundColor Blue
            & python scripts/cleanup_project.py --interactive
        }
        "5" {
            Write-Host "Exiting..." -ForegroundColor Gray
            exit 0
        }
        default {
            Write-Host "Invalid choice. Please try again." -ForegroundColor Red
        }
    }
    
    if ($choice -ne "5") {
        Write-Host ""
        Write-Host "Press any key to continue..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
} while ($choice -ne "5")
