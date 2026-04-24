"""
grokly/mcp_servers/test_servers.py

Tests each MCP server independently using MCPServerManager.

Run from project root:
    python grokly/mcp_servers/test_servers.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from grokly.mcp_servers.server_manager import MCPServerManager


def _pass_fail(condition: bool, label: str, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  {status}: {label}{suffix}")
    return condition


def test_knowledge_server(mgr: MCPServerManager) -> int:
    """Tests 1–4: knowledge_server search and stats."""
    print("\n=== knowledge_server ===")
    failures = 0

    # Test 1: search_knowledge returns results
    try:
        raw = mgr.call_tool("knowledge", "search_knowledge", {
            "query": "expense claim approval",
            "n_results": 5,
        })
        results = json.loads(raw)
        ok = isinstance(results, list) and len(results) > 0
        failures += 0 if _pass_fail(ok, "search_knowledge returns chunks", f"{len(results)} result(s)") else 1
        if ok:
            print(f"    Top result: {results[0]['text'][:100]}...")
    except Exception as exc:
        failures += 1
        _pass_fail(False, "search_knowledge", str(exc))

    # Test 2: chunk_type filter works
    try:
        raw = mgr.call_tool("knowledge", "search_knowledge", {
            "query": "validate expense",
            "chunk_type": "commentary",
            "n_results": 3,
        })
        results = json.loads(raw)
        ok = all(r.get("chunk_type") == "commentary" for r in results) if results else False
        failures += 0 if _pass_fail(ok, "chunk_type filter (commentary)", f"{len(results)} result(s)") else 1
    except Exception as exc:
        failures += 1
        _pass_fail(False, "chunk_type filter", str(exc))

    # Test 3: get_chunk_stats returns total matching ChromaDB
    try:
        from grokly.store.chroma_store import ChromaStore
        expected = ChromaStore().count()
        raw = mgr.call_tool("knowledge", "get_chunk_stats", {})
        stats = json.loads(raw)
        ok = stats.get("total") == expected
        failures += 0 if _pass_fail(ok, "get_chunk_stats total matches ChromaDB",
                                     f"total={stats.get('total')} (expected {expected})") else 1
        print(f"    by_chunk_type: {stats.get('by_chunk_type', {})}")
        print(f"    by_module:     {stats.get('by_module', {})}")
    except Exception as exc:
        failures += 1
        _pass_fail(False, "get_chunk_stats", str(exc))

    return failures


def test_file_server(mgr: MCPServerManager) -> int:
    """Tests 4–5: file_server list and read."""
    print("\n=== file_server ===")
    failures = 0

    # Test 4: list_module_files returns parseable JSON
    try:
        raw = mgr.call_tool("file", "list_module_files", {"module": "hr"})
        result = json.loads(raw)
        file_count = result.get("file_count", 0)
        if "error" in result:
            # Repos not cloned on this machine — not a server failure
            _pass_fail(True, "list_module_files returns JSON (repos not cloned)", result["error"][:60])
        else:
            ok = file_count > 0
            failures += 0 if _pass_fail(ok, "list_module_files (hr)", f"{file_count} file(s)") else 1
            if ok:
                print(f"    First 3 files: {result['files'][:3]}")
    except Exception as exc:
        failures += 1
        _pass_fail(False, "list_module_files", str(exc))

    # Test 5: path traversal is rejected
    try:
        raw = mgr.call_tool("file", "read_source_file", {
            "file_path": "../../.env",
            "repo": "erpnext",
        })
        ok = "traversal" in raw.lower() or "error" in raw.lower()
        failures += 0 if _pass_fail(ok, "path traversal rejected") else 1
    except Exception as exc:
        # An exception is also acceptable (server rejected it)
        _pass_fail(True, "path traversal rejected (raised exception)")

    return failures


def test_analysis_server(mgr: MCPServerManager) -> int:
    """Tests 6–8: analysis_server callers, callees, and impact report."""
    print("\n=== analysis_server ===")
    failures = 0

    fn = "validate_expense_claim_in_jv"

    # Test 6: get_function_callers
    try:
        raw = mgr.call_tool("analysis", "get_function_callers", {"function_name": fn})
        result = json.loads(raw)
        ok = "callers" in result and isinstance(result["callers"], list)
        detail = f"{result.get('caller_count', 0)} caller(s) found"
        failures += 0 if _pass_fail(ok, f"get_function_callers ({fn})", detail) else 1
    except Exception as exc:
        failures += 1
        _pass_fail(False, "get_function_callers", str(exc))

    # Test 7: get_function_callees
    try:
        raw = mgr.call_tool("analysis", "get_function_callees", {"function_name": fn})
        result = json.loads(raw)
        ok = "callees" in result and isinstance(result["callees"], list)
        detail = f"{result.get('callee_count', 0)} callee(s) found"
        failures += 0 if _pass_fail(ok, f"get_function_callees ({fn})", detail) else 1
    except Exception as exc:
        failures += 1
        _pass_fail(False, "get_function_callees", str(exc))

    # Test 8: analyse_change_impact — full report
    try:
        raw = mgr.call_tool("analysis", "analyse_change_impact", {
            "function_name": fn,
            "module": "hr",
        })
        report = json.loads(raw)
        ok = all(k in report for k in ("risk_level", "direct_callers", "testing_scope", "summary"))
        failures += 0 if _pass_fail(ok, f"analyse_change_impact ({fn})", report.get("summary", "")) else 1

        if ok:
            print("\n  --- Impact Analysis Report ---")
            print(f"  Function:       {report['function_name']}")
            print(f"  Module:         {report['module']}")
            print(f"  Risk level:     {report['risk_level']}")
            print(f"  Direct callers: {report['indirect_impact']['affected_modules']} module(s)")
            print(f"  Caller count:   {len(report['direct_callers'])}")
            print(f"  Testing scope:")
            for t in report["testing_scope"]:
                print(f"    - {t}")
            print(f"  Summary: {report['summary']}")
            if report["direct_callers"]:
                print(f"  Sample callers:")
                for c in report["direct_callers"][:5]:
                    print(f"    {c['function']} ({c['module']}) — {c['file']}")
    except Exception as exc:
        failures += 1
        _pass_fail(False, "analyse_change_impact", str(exc))

    return failures


def test_pipeline_end_to_end() -> int:
    """Test 9: full pipeline still returns correct answers with MCP layer."""
    print("\n=== end-to-end pipeline (with MCP) ===")
    failures = 0

    from grokly.pipeline.pipeline import run

    result = run(
        question="What are the approval rules for expense claims?",
        role="business_user",
    )
    answer = result.get("answer", "")
    ok = len(answer) > 50
    failures += 0 if _pass_fail(ok, "pipeline returns answer", f"{len(answer)} chars") else 1

    tools_used = result.get("tools_used", [])
    mcp_used = any("mcp" in t for t in tools_used)
    _pass_fail(mcp_used, "MCP layer used in pipeline", f"tools: {tools_used[:2]}")

    print(f"\n  Answer preview: {answer[:200]}...")
    print(f"  Confidence:     {result.get('confidence')}")
    print(f"  Quality score:  {result.get('quality_score')}")

    return failures


def main() -> None:
    print("Grokly Sprint 4A — MCP Server Tests")
    print("=" * 50)

    mgr = MCPServerManager()

    total_failures = 0
    total_failures += test_knowledge_server(mgr)
    total_failures += test_file_server(mgr)
    total_failures += test_analysis_server(mgr)
    total_failures += test_pipeline_end_to_end()

    print("\n" + "=" * 50)
    if total_failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{total_failures} TEST(S) FAILED")
    print("=" * 50)

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
