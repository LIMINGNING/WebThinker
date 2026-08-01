param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("baseline", "optimized")]
  [string]$Mode,
  [int]$Seed = 20260727,
  [int]$MaxSearchLimit = 15,
  [int]$TopK = 5,
  [int]$MaxTokens = 4096,
  [int]$RequestTimeout = 300,
  [int]$SequenceTimeout = 0,
  [int]$FinalAnswerReserveSeconds = 0,
  [int]$FinalAnswerMaxTokens = 4096,
  [int]$GenerationRetryLimit = 2,
  [int]$EvidenceMaxRecords = 8,
  [int]$EvidenceFollowupLimit = 4,
  [int]$EvidenceStagnationLimit = 2,
  [int]$EvidenceMinCompleteRounds = 2,
  [string]$IncludeIds = "",
  [string]$PythonPath = "",
  [string]$CacheDir = "",
  [switch]$EvidenceAdvisoryComplete,
  [switch]$EvidenceGlobalRequirements,
  [switch]$EvidenceRiskTiers,
  [switch]$DisableLeakageFilter,
  [switch]$DryRun
)

if (-not $env:DEEPSEEK_API_KEY) {
  throw "Please set DEEPSEEK_API_KEY in the current PowerShell session."
}
if (-not $env:SERPER_API_KEY) {
  throw "Please set SERPER_API_KEY in the current PowerShell session."
}

if (-not $PythonPath) {
  $localPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
  $siblingPython = Join-Path (Split-Path $PSScriptRoot -Parent) "webthinker\.venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $localPython) {
    $PythonPath = $localPython
  } elseif (Test-Path -LiteralPath $siblingPython) {
    $PythonPath = $siblingPython
  } else {
    throw "Could not find the WebThinker virtual-environment Python executable."
  }
}

if (-not $CacheDir) {
  $CacheDir = Join-Path $PSScriptRoot "cache"
}
if ($EvidenceAdvisoryComplete -and $Mode -ne "optimized") {
  throw "EvidenceAdvisoryComplete is available only in optimized mode."
}
if ($EvidenceGlobalRequirements -and -not $EvidenceAdvisoryComplete) {
  throw "EvidenceGlobalRequirements requires EvidenceAdvisoryComplete."
}
if ($EvidenceRiskTiers -and -not $EvidenceGlobalRequirements) {
  throw "EvidenceRiskTiers requires EvidenceGlobalRequirements."
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"

$runArgs = @(
  "scripts\run_web_thinker.py",
  "--dataset_name", "gaia",
  "--split", "dev",
  "--subset_num", "-1",
  "--seed", "$Seed",
  "--concurrent_limit", "1",
  "--max_search_limit", "$MaxSearchLimit",
  "--top_k", "$TopK",
  "--temperature", "0",
  "--top_p", "1",
  "--max_tokens", "$MaxTokens",
  "--request_timeout", "$RequestTimeout",
  "--sequence_timeout", "$SequenceTimeout",
  "--generation_retry_limit", "$GenerationRetryLimit",
  "--search_engine", "serper",
  "--cache_dir", "$CacheDir",
  "--serper_api_key", "env:SERPER_API_KEY",
  "--api_base_url", "https://api.deepseek.com",
  "--aux_api_base_url", "https://api.deepseek.com",
  "--model_name", "deepseek-v4-pro",
  "--aux_model_name", "deepseek-v4-pro",
  "--api_key", "env:DEEPSEEK_API_KEY",
  "--aux_api_key", "env:DEEPSEEK_API_KEY",
  "--use_chat_completions",
  "--thinking", "disabled",
  "--final_answer_reserve_seconds", "$FinalAnswerReserveSeconds",
  "--final_answer_max_tokens", "$FinalAnswerMaxTokens"
)

if (-not $DisableLeakageFilter) {
  $runArgs += "--filter_benchmark_leakage"
}
if ($IncludeIds) {
  $runArgs += @("--include_ids", $IncludeIds)
}

if ($Mode -eq "optimized") {
  $evidenceVariant = if ($EvidenceRiskTiers) {
    "evidence-completeness-v2c-risk-tiers-no-total-timeout"
  } elseif ($EvidenceGlobalRequirements) {
    "evidence-completeness-v2b-global-requirements-no-total-timeout"
  } elseif ($EvidenceAdvisoryComplete) {
    "evidence-completeness-v2-advisory-complete-no-total-timeout"
  } else {
    "evidence-completeness-v1-no-total-timeout"
  }
  $experimentLabel = if ($DisableLeakageFilter) {
    "$evidenceVariant-fetch-fixed-unfiltered-full"
  } else {
    "$evidenceVariant-fetch-fixed-leak-filtered-full"
  }
  $runArgs += @(
    "--use_evidence_completeness",
    "--evidence_delivery_mode", "aux_summary",
    "--evidence_max_records", "$EvidenceMaxRecords",
    "--evidence_followup_limit", "$EvidenceFollowupLimit",
    "--evidence_stagnation_limit", "$EvidenceStagnationLimit",
    "--evidence_min_complete_rounds", "$EvidenceMinCompleteRounds",
    "--experiment_label", $experimentLabel
  )
  if ($EvidenceAdvisoryComplete) {
    $runArgs += "--evidence_advisory_complete"
  }
  if ($EvidenceGlobalRequirements) {
    $runArgs += "--evidence_global_requirements"
  }
  if ($EvidenceRiskTiers) {
    $runArgs += "--evidence_risk_tiers"
  }
} else {
  $experimentLabel = if ($DisableLeakageFilter) {
    "web-fetch-fallback-no-total-timeout-unfiltered-full-baseline"
  } else {
    "web-fetch-fallback-no-total-timeout-leak-filtered-full-baseline"
  }
  $runArgs += @(
    "--experiment_label", $experimentLabel
  )
}

$filterEnabled = -not $DisableLeakageFilter
Write-Output "Starting paired GAIA run: mode=$Mode, leakage_filter=$filterEnabled, seed=$Seed, max_search_limit=$MaxSearchLimit, include_ids=$IncludeIds, advisory_complete=$EvidenceAdvisoryComplete, global_requirements=$EvidenceGlobalRequirements, risk_tiers=$EvidenceRiskTiers"
if ($DryRun) {
  Write-Output ($runArgs -join " ")
  exit 0
}

& $PythonPath @runArgs
if ($LASTEXITCODE -ne 0) {
  throw "WebThinker run failed with exit code $LASTEXITCODE."
}
