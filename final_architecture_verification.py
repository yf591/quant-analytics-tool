#!/usr/bin/env python3
"""
Risk Management UI Architecture Final Verification
Check for any remaining hardcoded calculations in frontend files
"""

import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.risk.risk_metrics import RiskMetrics


def check_hardcoded_patterns_in_files():
    """Check for hardcoded risk calculation patterns in frontend files"""

    print("🔍 Checking for Hardcoded Risk Calculations...")

    # Files to check
    risk_management_files = [
        "streamlit_app/pages/06_risk_management.py",
        "streamlit_app/utils/risk_management_utils.py",
    ]

    # Patterns that indicate hardcoded calculations
    hardcoded_patterns = {
        # Volatility calculations
        r"\.std\(\)\s*\*\s*np\.sqrt\(252\)": "Direct volatility annualization",
        r"\.std\(\)\s*\*\s*math\.sqrt\(252\)": "Direct volatility annualization (math)",
        r"\*\s*252\*\*0\.5": "Direct volatility annualization (power)",
        # Return calculations
        r"\.mean\(\)\s*\*\s*252": "Direct return annualization",
        # Variance calculations
        r"\.var\(\)\s*\*\s*252": "Direct variance annualization",
        # VaR calculations
        r"\.quantile\(0\.05\)": "Direct VaR calculation (5%)",
        r"\.quantile\(0\.01\)": "Direct VaR calculation (1%)",
        # Sharpe ratio calculations
        r"[a-zA-Z_][a-zA-Z0-9_]*\s*/\s*[a-zA-Z_][a-zA-Z0-9_]*\s*if.*vol": "Direct Sharpe ratio calculation",
        # Common financial constants
        r"1\.96": "Z-score for 95% confidence",
        r"2\.33": "Z-score for 99% confidence",
    }

    issues_found = []

    for file_path in risk_management_files:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        print(f"\n📄 Checking {file_path}...")

        for pattern, description in hardcoded_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)

            for match in matches:
                # Find line number
                line_num = content[: match.start()].count("\n") + 1
                line_content = lines[line_num - 1].strip()

                # Skip if it's a comment or in test pattern list
                if (
                    line_content.startswith("#")
                    or "hardcoded_patterns" in line_content
                    or "test_" in file_path
                    or '"' in line_content
                    and pattern.replace("\\", "") in line_content
                ):
                    continue

                issues_found.append(
                    {
                        "file": file_path,
                        "line": line_num,
                        "pattern": pattern,
                        "description": description,
                        "code": line_content,
                    }
                )

                print(f"❌ Line {line_num}: {description}")
                print(f"   Code: {line_content}")

    return issues_found


def test_backend_integration():
    """Test that backend integration works correctly"""

    print("\n🧪 Testing Backend Integration...")

    try:
        # Generate sample returns data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 100), index=dates, name="returns"
        )

        # Test RiskMetrics backend
        risk_metrics = RiskMetrics(confidence_level=0.95)

        # Test risk-adjusted returns (this should provide annual_volatility, annual_return, etc.)
        risk_adj_metrics = risk_metrics.risk_adjusted_returns(returns)

        required_metrics = [
            "annual_volatility",
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
        ]
        missing_metrics = []

        for metric in required_metrics:
            if metric not in risk_adj_metrics:
                missing_metrics.append(metric)
            else:
                print(f"✅ {metric}: {risk_adj_metrics[metric]:.4f}")

        if missing_metrics:
            print(f"❌ Missing backend metrics: {missing_metrics}")
            return False

        # Test VaR calculation
        var_result = risk_metrics.value_at_risk(returns, method="historical")
        print(f"✅ VaR (Historical): {var_result:.4f}")

        # Test CVaR calculation
        cvar_result = risk_metrics.conditional_var(returns, method="historical")
        print(f"✅ CVaR (Historical): {cvar_result:.4f}")

        return True

    except Exception as e:
        print(f"❌ Backend integration error: {e}")
        return False


def check_proper_backend_usage():
    """Check that frontend files properly use backend methods"""

    print("\n🔧 Checking Backend Usage Patterns...")

    risk_management_files = [
        "streamlit_app/pages/06_risk_management.py",
        "streamlit_app/utils/risk_management_utils.py",
    ]

    # Good patterns that should be present
    good_patterns = {
        r"from src\.risk\.risk_metrics import RiskMetrics": "Proper backend import",
        r"risk_metrics\.risk_adjusted_returns\(": "Using backend risk-adjusted returns",
        r"risk_metrics\.value_at_risk\(": "Using backend VaR calculation",
        r"risk_metrics\.conditional_var\(": "Using backend CVaR calculation",
        r"\.get\(['\"]annual_volatility['\"]": "Getting volatility from backend",
        r"\.get\(['\"]annual_return['\"]": "Getting return from backend",
        r"\.get\(['\"]sharpe_ratio['\"]": "Getting Sharpe ratio from backend",
    }

    usage_found = {}

    for file_path in risk_management_files:
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"\n📄 Checking {file_path}...")
        file_usage = {}

        for pattern, description in good_patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            file_usage[description] = len(matches)

            if matches:
                print(f"✅ {description}: {len(matches)} usage(s)")
            else:
                print(f"⚠️  {description}: Not found")

        usage_found[file_path] = file_usage

    return usage_found


def main():
    """Run comprehensive architecture verification"""

    print("🚀 Risk Management UI Architecture Final Verification")
    print("=" * 70)

    # 1. Check for hardcoded patterns
    issues = check_hardcoded_patterns_in_files()

    # 2. Test backend integration
    backend_ok = test_backend_integration()

    # 3. Check proper backend usage
    usage_patterns = check_proper_backend_usage()

    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL VERIFICATION SUMMARY")
    print("=" * 70)

    # Hardcoded calculation results
    if issues:
        print(f"❌ HARDCODED CALCULATIONS FOUND: {len(issues)} issues")
        print("\nIssues found:")
        for issue in issues:
            print(f"  • {issue['file']}:{issue['line']} - {issue['description']}")
    else:
        print("✅ NO HARDCODED CALCULATIONS FOUND")

    # Backend integration results
    print(f"\nBackend Integration: {'✅ WORKING' if backend_ok else '❌ FAILED'}")

    # Usage pattern results
    print("\nBackend Usage Summary:")
    for file_path, patterns in usage_patterns.items():
        file_name = file_path.split("/")[-1]
        total_usage = sum(patterns.values())
        print(f"  • {file_name}: {total_usage} backend method calls")

    # Final verdict
    if not issues and backend_ok:
        print(f"\n🎉 ARCHITECTURE VERIFICATION: ✅ PASSED")
        print("   • No hardcoded calculations in frontend files")
        print("   • Backend integration working correctly")
        print("   • Proper separation of concerns maintained")
        return True
    else:
        print(f"\n⚠️  ARCHITECTURE VERIFICATION: ❌ FAILED")
        if issues:
            print("   • Hardcoded calculations still present in frontend")
        if not backend_ok:
            print("   • Backend integration issues detected")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
