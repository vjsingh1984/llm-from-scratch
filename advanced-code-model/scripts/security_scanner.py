#!/usr/bin/env python3
"""
Security Scanner for Generated Code
Scans generated code for potential security vulnerabilities
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityIssue:
    """Represents a security issue found in code"""
    issue_type: str
    severity: Severity
    line_number: int
    line_content: str
    description: str
    recommendation: str

class CodeSecurityScanner:
    """
    Scans generated code for security vulnerabilities
    Based on OWASP Top 10 and common code security issues
    """

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> List[Dict]:
        """Build regex patterns for vulnerability detection"""
        return [
            # ============================================================
            # INJECTION VULNERABILITIES
            # ============================================================
            {
                "pattern": r"os\.system\s*\(",
                "type": "command_injection",
                "severity": Severity.CRITICAL,
                "description": "Potential command injection via os.system()",
                "recommendation": "Use subprocess with shell=False and proper argument handling"
            },
            {
                "pattern": r"subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True",
                "type": "command_injection",
                "severity": Severity.CRITICAL,
                "description": "Shell=True enables command injection",
                "recommendation": "Use shell=False and pass arguments as a list"
            },
            {
                "pattern": r"\beval\s*\(",
                "type": "code_injection",
                "severity": Severity.CRITICAL,
                "description": "eval() can execute arbitrary code",
                "recommendation": "Avoid eval(). Use ast.literal_eval() for safe parsing"
            },
            {
                "pattern": r"\bexec\s*\(",
                "type": "code_injection",
                "severity": Severity.CRITICAL,
                "description": "exec() can execute arbitrary code",
                "recommendation": "Avoid exec(). Refactor to avoid dynamic code execution"
            },

            # ============================================================
            # SQL INJECTION
            # ============================================================
            {
                "pattern": r"execute\s*\(\s*[\"'].*%[sd].*[\"']\s*%",
                "type": "sql_injection",
                "severity": Severity.CRITICAL,
                "description": "String formatting in SQL query enables injection",
                "recommendation": "Use parameterized queries with placeholders"
            },
            {
                "pattern": r"execute\s*\(\s*f[\"'].*\{.*\}",
                "type": "sql_injection",
                "severity": Severity.CRITICAL,
                "description": "f-string in SQL query enables injection",
                "recommendation": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id=?', (user_id,))"
            },
            {
                "pattern": r"\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)",
                "type": "sql_injection",
                "severity": Severity.HIGH,
                "description": ".format() in SQL query may enable injection",
                "recommendation": "Use parameterized queries instead of string formatting"
            },

            # ============================================================
            # PATH TRAVERSAL
            # ============================================================
            {
                "pattern": r"\.\./\.\./",
                "type": "path_traversal",
                "severity": Severity.HIGH,
                "description": "Potential path traversal pattern detected",
                "recommendation": "Validate and sanitize file paths. Use os.path.realpath() to resolve paths"
            },
            {
                "pattern": r"open\s*\([^)]*\+[^)]*user",
                "type": "path_traversal",
                "severity": Severity.HIGH,
                "description": "User input in file path may enable path traversal",
                "recommendation": "Validate paths against a whitelist or use os.path.join() with sanitized input"
            },

            # ============================================================
            # HARDCODED SECRETS
            # ============================================================
            {
                "pattern": r"(?:password|passwd|pwd)\s*=\s*[\"'][^\"']{4,}[\"']",
                "type": "hardcoded_secret",
                "severity": Severity.HIGH,
                "description": "Hardcoded password detected",
                "recommendation": "Use environment variables or a secrets manager"
            },
            {
                "pattern": r"(?:api_key|apikey|api-key)\s*=\s*[\"'][^\"']{8,}[\"']",
                "type": "hardcoded_secret",
                "severity": Severity.HIGH,
                "description": "Hardcoded API key detected",
                "recommendation": "Use environment variables: os.environ.get('API_KEY')"
            },
            {
                "pattern": r"(?:secret|token)\s*=\s*[\"'][^\"']{8,}[\"']",
                "type": "hardcoded_secret",
                "severity": Severity.MEDIUM,
                "description": "Potential hardcoded secret or token",
                "recommendation": "Use environment variables or a secrets manager"
            },
            {
                "pattern": r"(?:aws_access_key|aws_secret|private_key)\s*=\s*[\"']",
                "type": "hardcoded_secret",
                "severity": Severity.CRITICAL,
                "description": "AWS/cloud credentials may be hardcoded",
                "recommendation": "Use IAM roles or environment variables for cloud credentials"
            },

            # ============================================================
            # INSECURE DESERIALIZATION
            # ============================================================
            {
                "pattern": r"pickle\.loads?\s*\(",
                "type": "insecure_deserialization",
                "severity": Severity.HIGH,
                "description": "pickle can execute arbitrary code during deserialization",
                "recommendation": "Use JSON or other safe formats. If pickle is required, only unpickle trusted data"
            },
            {
                "pattern": r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader\s*=\s*yaml\.SafeLoader)",
                "type": "insecure_deserialization",
                "severity": Severity.HIGH,
                "description": "yaml.load() without SafeLoader can execute code",
                "recommendation": "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)"
            },
            {
                "pattern": r"marshal\.loads?\s*\(",
                "type": "insecure_deserialization",
                "severity": Severity.MEDIUM,
                "description": "marshal can be unsafe with untrusted data",
                "recommendation": "Avoid marshal for untrusted data. Use JSON instead"
            },

            # ============================================================
            # CRYPTOGRAPHIC ISSUES
            # ============================================================
            {
                "pattern": r"hashlib\.md5\s*\(",
                "type": "weak_crypto",
                "severity": Severity.MEDIUM,
                "description": "MD5 is cryptographically broken",
                "recommendation": "Use SHA-256 or better: hashlib.sha256()"
            },
            {
                "pattern": r"hashlib\.sha1\s*\(",
                "type": "weak_crypto",
                "severity": Severity.MEDIUM,
                "description": "SHA1 is deprecated for security use",
                "recommendation": "Use SHA-256 or better: hashlib.sha256()"
            },
            {
                "pattern": r"random\.(random|randint|choice|shuffle)\s*\(",
                "type": "weak_crypto",
                "severity": Severity.LOW,
                "description": "random module is not cryptographically secure",
                "recommendation": "For security-sensitive use, use secrets module instead"
            },

            # ============================================================
            # NETWORK SECURITY
            # ============================================================
            {
                "pattern": r"verify\s*=\s*False",
                "type": "ssl_bypass",
                "severity": Severity.HIGH,
                "description": "SSL certificate verification disabled",
                "recommendation": "Always verify SSL certificates: verify=True"
            },
            {
                "pattern": r"ssl\._create_unverified_context",
                "type": "ssl_bypass",
                "severity": Severity.HIGH,
                "description": "Unverified SSL context creation",
                "recommendation": "Use ssl.create_default_context() for secure connections"
            },
            {
                "pattern": r"http://(?!localhost|127\.0\.0\.1)",
                "type": "insecure_transport",
                "severity": Severity.MEDIUM,
                "description": "HTTP (non-HTTPS) URL detected",
                "recommendation": "Use HTTPS for all external communications"
            },

            # ============================================================
            # XSS/HTML INJECTION
            # ============================================================
            {
                "pattern": r"innerHTML\s*=",
                "type": "xss",
                "severity": Severity.HIGH,
                "description": "innerHTML can enable XSS attacks",
                "recommendation": "Use textContent or sanitize input with DOMPurify"
            },
            {
                "pattern": r"document\.write\s*\(",
                "type": "xss",
                "severity": Severity.HIGH,
                "description": "document.write can enable XSS",
                "recommendation": "Use DOM methods like createElement() instead"
            },
            {
                "pattern": r"\.html\s*\([^)]*user|\.html\s*\([^)]*input",
                "type": "xss",
                "severity": Severity.HIGH,
                "description": "User input in .html() may enable XSS",
                "recommendation": "Use .text() or sanitize input before using .html()"
            },

            # ============================================================
            # DANGEROUS OPERATIONS
            # ============================================================
            {
                "pattern": r"os\.chmod\s*\([^)]*0?777",
                "type": "insecure_permissions",
                "severity": Severity.MEDIUM,
                "description": "Setting overly permissive file permissions (777)",
                "recommendation": "Use minimum necessary permissions (e.g., 0o644 for files)"
            },
            {
                "pattern": r"rm\s+-rf\s+/",
                "type": "dangerous_operation",
                "severity": Severity.CRITICAL,
                "description": "Dangerous rm -rf / command",
                "recommendation": "Never use rm -rf on root. Add safety checks"
            },
            {
                "pattern": r"DROP\s+(?:TABLE|DATABASE)",
                "type": "dangerous_operation",
                "severity": Severity.HIGH,
                "description": "DROP TABLE/DATABASE statement detected",
                "recommendation": "Ensure proper authorization and backup before destructive operations"
            },

            # ============================================================
            # LOGGING ISSUES
            # ============================================================
            {
                "pattern": r"print\s*\([^)]*password|print\s*\([^)]*secret|print\s*\([^)]*token",
                "type": "sensitive_data_exposure",
                "severity": Severity.MEDIUM,
                "description": "Potential logging of sensitive data",
                "recommendation": "Never log passwords, secrets, or tokens"
            },
            {
                "pattern": r"logging\.[^(]+\([^)]*password|logging\.[^(]+\([^)]*secret",
                "type": "sensitive_data_exposure",
                "severity": Severity.MEDIUM,
                "description": "Logging sensitive data",
                "recommendation": "Mask sensitive data in logs: password='***'"
            },
        ]

    def scan(self, code: str) -> List[SecurityIssue]:
        """
        Scan code for security vulnerabilities

        Args:
            code: Source code to scan

        Returns:
            List of SecurityIssue objects
        """
        issues = []
        lines = code.split('\n')

        for line_num, line in enumerate(lines, 1):
            for pattern_def in self.patterns:
                if re.search(pattern_def["pattern"], line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_type=pattern_def["type"],
                        severity=pattern_def["severity"],
                        line_number=line_num,
                        line_content=line.strip(),
                        description=pattern_def["description"],
                        recommendation=pattern_def["recommendation"]
                    ))

        return issues

    def scan_file(self, filepath: str) -> List[SecurityIssue]:
        """Scan a file for security vulnerabilities"""
        with open(filepath, 'r') as f:
            return self.scan(f.read())

    def is_safe(self, code: str) -> bool:
        """Return True if no security issues found"""
        return len(self.scan(code)) == 0

    def get_severity_counts(self, issues: List[SecurityIssue]) -> Dict[Severity, int]:
        """Count issues by severity"""
        counts = {s: 0 for s in Severity}
        for issue in issues:
            counts[issue.severity] += 1
        return counts

    def format_report(self, issues: List[SecurityIssue]) -> str:
        """Format issues as a readable report"""
        if not issues:
            return "No security issues found."

        report = ["=" * 60]
        report.append("SECURITY SCAN REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary
        counts = self.get_severity_counts(issues)
        report.append("SUMMARY:")
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            if counts[severity] > 0:
                report.append(f"  {severity.value.upper()}: {counts[severity]}")
        report.append("")

        # Details
        report.append("ISSUES:")
        report.append("-" * 60)

        for i, issue in enumerate(issues, 1):
            report.append(f"\n[{i}] {issue.severity.value.upper()}: {issue.issue_type}")
            report.append(f"    Line {issue.line_number}: {issue.line_content[:60]}...")
            report.append(f"    Description: {issue.description}")
            report.append(f"    Fix: {issue.recommendation}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


class PromptSanitizer:
    """Sanitize user prompts to prevent prompt injection"""

    BLOCKED_PATTERNS = [
        r'ignore\s+(previous|all|prior)\s+(instructions|prompts)',
        r'disregard\s+(previous|all|prior)',
        r'forget\s+(previous|all|prior)',
        r'system\s*:\s*',
        r'<\|.*\|>',  # Special tokens
        r'\[INST\]',  # Instruction markers
        r'\[/INST\]',
        r'<<SYS>>',
        r'<</SYS>>',
        r'Human:',
        r'Assistant:',
    ]

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length

    def sanitize(self, prompt: str) -> str:
        """Remove potentially malicious content from prompts"""
        sanitized = prompt

        for pattern in self.BLOCKED_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        # Limit length
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]

        return sanitized.strip()

    def is_safe(self, prompt: str) -> bool:
        """Check if prompt contains any blocked patterns"""
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False
        return True


# ============================================================
# CLI Interface
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scan code for security vulnerabilities")
    parser.add_argument("file", nargs="?", help="File to scan (or reads from stdin)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    scanner = CodeSecurityScanner()

    if args.file:
        issues = scanner.scan_file(args.file)
    else:
        import sys
        code = sys.stdin.read()
        issues = scanner.scan(code)

    if args.json:
        import json
        output = {
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity.value,
                    "line": i.line_number,
                    "description": i.description,
                    "recommendation": i.recommendation
                }
                for i in issues
            ],
            "summary": {s.value: c for s, c in scanner.get_severity_counts(issues).items()}
        }
        print(json.dumps(output, indent=2))
    else:
        print(scanner.format_report(issues))

    # Exit with error if critical/high issues found
    counts = scanner.get_severity_counts(issues)
    if counts[Severity.CRITICAL] > 0 or counts[Severity.HIGH] > 0:
        exit(1)


if __name__ == "__main__":
    main()
