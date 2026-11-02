# sign_exe.ps1 -- sign FlyMorph.exe as "Stark Huang" with a local certificate.
# Order matters: export the .cer BEFORE importing it into the trust stores.

param(
    [string]$Exe = (Join-Path $PSScriptRoot "..\build\dist\FlyMorph\FlyMorph.exe"),
    [string]$CertDir = (Join-Path $PSScriptRoot "..\build\certs"),
    [string]$Subject = "CN=Stark Huang",
    [string]$OldSubject = "CN=FlyMorph Local Signing"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $CertDir | Out-Null

# drop the previous certificate (old publisher name) from every store we touched
foreach ($store in @("Cert:\CurrentUser\My", "Cert:\CurrentUser\Root",
                     "Cert:\CurrentUser\TrustedPublisher")) {
    Get-ChildItem $store -ErrorAction SilentlyContinue |
        Where-Object { $_.Subject -eq $OldSubject } |
        ForEach-Object {
            try { Remove-Item $_.PSPath -Force; Write-Host "removed old cert from $store" } catch { }
        }
}

$pfx = Join-Path $CertDir "flymorph-signing.pfx"
$cer = Join-Path $CertDir "flymorph-signing.cer"

$cert = Get-ChildItem Cert:\CurrentUser\My |
        Where-Object { $_.Subject -eq $Subject -and $_.HasPrivateKey } |
        Sort-Object NotAfter -Descending | Select-Object -First 1
if (-not $cert) {
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $Subject `
        -KeyUsage DigitalSignature -KeyLength 2048 -NotAfter (Get-Date).AddYears(5) `
        -CertStoreLocation "Cert:\CurrentUser\My"
    Write-Host "created certificate: $($cert.Thumbprint)"
} else {
    Write-Host "reusing certificate: $($cert.Thumbprint)"
}

Export-Certificate -Cert $cert -FilePath $cer -Force | Out-Null
Export-PfxCertificate -Cert $cert -FilePath $pfx `
    -Password (ConvertTo-SecureString "flymorph" -Force -AsPlainText) | Out-Null
Write-Host "exported: $cer"

foreach ($store in @("Cert:\CurrentUser\Root", "Cert:\CurrentUser\TrustedPublisher")) {
    try {
        Import-Certificate -FilePath $cer -CertStoreLocation $store | Out-Null
        Write-Host "trusted in $store"
    } catch {
        Write-Host "could not import into ${store}: $($_.Exception.Message)"
    }
}

if (Test-Path $Exe) {
    $sig = Set-AuthenticodeSignature -FilePath $Exe -Certificate $cert
    Write-Host "signature: $($sig.Status)  $($sig.StatusMessage)"
    try { Unblock-File -Path $Exe } catch { }
} else {
    Write-Host "exe not found: $Exe"
}
