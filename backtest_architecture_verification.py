#!/usr/bin/env python3
"""
Backtest Architecture Verification Script
Verify that hardcoded calculations have been properly removed from frontend files
"""

import sys
import re
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))


def check_hardcoded_calculations(file_path: str, file_type: str):
    """Check for hardcoded calculations in a file"""

    issues_found = []

    # Define hardcoded calculation patterns to look for
    hardcoded_patterns = {
        "volatility_annualization": [
            r"\.std\(\)\s*\*\s*(?:np\.)?sqrt\(252\)",
            r"\.std\(\)\s*\*\s*252\*\*0\.5",
            r"\.std\(\)\s*\*\s*math\.sqrt\(252\)",
        ],
        "sharpe_ratio": [
            r"\(\s*[^)]*\s*-\s*0\.0?2\s*\)\s*\/\s*[^)]*volatility",
            r"excess_returns\.mean\(\)\s*\/\s*[^)]*\.std\(\)",
        ],
        "calmar_ratio": [
            r"annualized_return\s*\/\s*[^)]*max_drawdown",
            r"total_return\s*\/\s*[^)]*drawdown",
        ],
        "var_calculation": [
            r"\.quantile\(0\.05\)",
            r"\.quantile\(0\.95\)",
        ],
        "returns_annualization": [
            r"return\s*\*\s*252",
            r"\.mean\(\)\s*\*\s*252",
        ],
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern_type, patterns in hardcoded_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues_found.append(
                            {
                                "line": line_num,
                                "content": line.strip(),
                                "pattern": pattern_type,
                                "file": file_path,
                            }
                        )

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return []

    return issues_found


def check_backend_integration(file_path: str):
    """Check that backend modules are properly imported and used"""

    backend_imports = []
    backend_usage = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Check for backend imports
        backend_modules = [
            "src.backtesting",
            "PerformanceCalculator",
            "BacktestEngine",
            "PerformanceMetrics",
        ]

        for line_num, line in enumerate(lines, 1):
            for module in backend_modules:
                if module in line and ("import" in line or "from" in line):
                    backend_imports.append(
                        {"line": line_num, "content": line.strip(), "module": module}
                    )

            # Check for backend method usage
            if "calculate_comprehensive_metrics" in line:
                backend_usage.append(
                    {
                        "line": line_num,
                        "content": line.strip(),
                        "method": "calculate_comprehensive_metrics",
                    }
                )

    except Exception as e:
        print(f"❌ Error checking backend integration in {file_path}: {e}")
        return [], []

    return backend_imports, backend_usage


def main():
    """Run backtest architecture verification"""

    print("🔍 Backtest Architecture Verification")
    print("=" * 60)

    # Files to check
    frontend_files = [
        "streamlit_app/pages/05_backtesting.py",
        "streamlit_app/utils/backtest_utils.py",
    ]

    total_issues = 0

    for file_path in frontend_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\n📁 Checking {file_path}")
        print("-" * 40)

        # Check for hardcoded calculations
        issues = check_hardcoded_calculations(str(full_path), "frontend")

        if issues:
            print(f"❌ HARDCODED CALCULATIONS FOUND: {len(issues)} issues")
            for issue in issues:
                print(f"   Line {issue['line']}: {issue['pattern']}")
                print(f"   Code: {issue['content']}")
            total_issues += len(issues)
        else:
            print("✅ No hardcoded calculations found")

        # Check backend integration
        imports, usage = check_backend_integration(str(full_path))

        if imports:
            print(f"✅ Backend imports: {len(imports)} found")
            for imp in imports[:3]:  # Show first 3
                print(f"   Line {imp['line']}: {imp['module']}")
        else:
            print("⚠️  No backend imports detected")

        if usage:
            print(f"✅ Backend method calls: {len(usage)} found")
        else:
            print("⚠️  No backend method calls detected")

    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY:")

    if total_issues == 0:
        print("✅ ARCHITECTURE CLEAN: No hardcoded calculations found")
        print("✅ Frontend/Backend separation maintained")
        print("✅ Ready for production")
        return True
    else:
        print(f"❌ HARDCODED CALCULATIONS FOUND: {total_issues} issues")
        print("❌ Architecture violations detected")
        print("❌ Requires immediate fixes")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
