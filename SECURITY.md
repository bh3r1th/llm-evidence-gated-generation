# Security Policy

## Supported Versions

EGA is currently in an early release stage.

- **v1.0 (initial release)** — Best-effort security support
- Only the **latest tagged version** is expected to receive updates
- Older versions may not receive patches

Security guarantees are limited at this stage and will improve as the project matures.

---

## Reporting a Vulnerability

Please report vulnerabilities **privately**. Do not open public issues for undisclosed vulnerabilities.

Contact:
- **Bharath Nunepalli**
- Email: bn3020@protonmail.com

If possible, use **GitHub Security Advisories** for coordinated disclosure.

---

## What to Include

Please provide as much detail as possible:

- Clear description of the vulnerability
- Steps to reproduce (code, inputs, configuration)
- Affected components/files
- Impact assessment:
  - Confidentiality
  - Integrity
  - Availability
- Any known mitigations or workarounds

---

## Scope

This project is a **runtime verification layer** and does not directly manage:

- Authentication / authorization
- Data storage or encryption
- Network security

However, vulnerabilities related to:
- Incorrect enforcement decisions
- Bypass of verification logic
- Unsafe output propagation
- Dependency risks

are considered in scope.

---

## Response Expectations

- Reports will be acknowledged and triaged
- Fixes are **best-effort**
- No guaranteed SLA at this stage
- Critical issues will be prioritized when feasible

---

## Disclosure Policy

- Please allow reasonable time for investigation and patching before public disclosure
- Once resolved, fixes may be released with minimal public detail initially
- Full disclosure may follow after users have had time to upgrade

---

## Notes

EGA is evolving rapidly. Security practices and guarantees will be strengthened in future releases as real-world usage increases.