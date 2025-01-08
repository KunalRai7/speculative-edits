$API_URL = "https://speculative-edits-production.up.railway.app"

function Test-APIEndpoint {
    param (
        [string]$TestName,
        [hashtable]$Body
    )
    
    Write-Host "`n$TestName..." -ForegroundColor Cyan
    try {
        $jsonBody = $Body | ConvertTo-Json
        Write-Host "Request Body:" -ForegroundColor Gray
        Write-Host $jsonBody -ForegroundColor Gray
        
        $result = Invoke-RestMethod -Uri "$API_URL/edit" `
            -Method Post `
            -ContentType "application/json" `
            -Body $jsonBody `
            -ErrorAction Stop

        Write-Host "Response:" -ForegroundColor Green
        $result | ConvertTo-Json -Depth 10
    }
    catch {
        Write-Host "Error occurred:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $reader.BaseStream.Position = 0
            $reader.DiscardBufferedData()
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response body:" -ForegroundColor Red
            Write-Host $responseBody -ForegroundColor Red
        }
    }
}

# Test Speculative Method
Test-APIEndpoint -TestName "Testing Speculative Method" -Body @{
    method = "speculative"
}

# Test Vanilla Method
Test-APIEndpoint -TestName "Testing Vanilla Method" -Body @{
    method = "vanilla"
}

# Test with Custom Parameters
Test-APIEndpoint -TestName "Testing with Custom Parameters" -Body @{
    method = "speculative"
    max_tokens = 500
} 