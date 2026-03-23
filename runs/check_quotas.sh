#!/bin/bash
# Check AWS GPU quota request status across all regions.
# Usage: bash runs/check_quotas.sh

REGIONS="eu-west-2 us-east-1 us-west-2 eu-north-1 eu-south-1 eu-central-1 eu-west-1 ap-northeast-1 ap-southeast-1"

echo "=== AWS GPU Quota Status — $(date '+%Y-%m-%d %H:%M') ==="
echo ""
printf "%-16s  %-12s %-12s %-12s %-12s\n" "REGION" "P SPOT" "P ON-DEM" "G/VT SPOT" "G/VT ON-DEM"
printf "%-16s  %-12s %-12s %-12s %-12s\n" "────────────────" "────────────" "────────────" "────────────" "────────────"

for region in $REGIONS; do
  p_spot=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-7212CCBC --region $region --query 'Quota.Value' --output text 2>/dev/null || echo "N/A")
  p_od=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-417A185B --region $region --query 'Quota.Value' --output text 2>/dev/null || echo "N/A")
  g_spot=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-3819A6DF --region $region --query 'Quota.Value' --output text 2>/dev/null || echo "N/A")
  g_od=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region $region --query 'Quota.Value' --output text 2>/dev/null || echo "N/A")
  printf "%-16s  %-12s %-12s %-12s %-12s\n" "$region" "$p_spot" "$p_od" "$g_spot" "$g_od"
done

echo ""
echo "=== Pending Requests ==="
echo ""

for region in $REGIONS; do
  pending=$(aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region $region \
    --query 'RequestedQuotas[?Status!=`CASE_CLOSED`]' --output json 2>/dev/null)
  count=$(echo "$pending" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
  if [ "$count" -gt 0 ] 2>/dev/null; then
    echo "[$region]"
    echo "$pending" | python3 -c "
import sys, json
for r in json.load(sys.stdin):
    print(f'  {r[\"QuotaName\"]:<42s} {r[\"Status\"]:<14s} {r[\"DesiredValue\"]:.0f}')
" 2>/dev/null
  fi
done

echo ""
echo "Target: P Spot >= 192 vCPUs (p5.48xlarge = 8x H100)"
