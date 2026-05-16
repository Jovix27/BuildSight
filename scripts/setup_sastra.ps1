Write-Host "--- SASTRA Supercomputer Setup ---" -ForegroundColor Cyan
Write-Host "1. Connect to Gateway: joseva@172.16.13.62"
Write-Host "   Password: sastra@2026"
Write-Host "2. Once connected, run: ssh node1"
Write-Host "3. Then run: nvidia-smi"
Write-Host "4. Finally: source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate"
Write-Host "----------------------------------"

ssh -o StrictHostKeyChecking=no joseva@172.16.13.62
