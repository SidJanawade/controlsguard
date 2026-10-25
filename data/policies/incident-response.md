# Incident Response Policy

This policy defines how technology incidents are classified, escalated, reviewed and, where required, reported to regulators.

## C-IR-01 Severity classification and 15-minute Sev1 escalation
- Owner: Head of Service Delivery
- Frequency: Per incident
- Risks: R-OUTAGE
- Systems: payments-gateway, ledger-core, trade-db, prod-servers
- Related: C-IR-02, C-CM-02
- Evidence: Incident ticket timeline

Every incident is given a severity of Sev1 to Sev4 within 10 minutes of detection. A Sev1 incident, meaning a customer-facing outage or confirmed data loss, must be escalated to the incident commander and senior management within 15 minutes.

## C-IR-02 Post-incident review within 5 business days
- Owner: Head of Service Delivery
- Frequency: Per Sev1 or Sev2 incident
- Risks: R-OUTAGE
- Systems: payments-gateway, ledger-core, trade-db, prod-servers
- Related: C-IR-01
- Evidence: Blameless post-incident review document

A blameless post-incident review is held within 5 business days of every Sev1 and Sev2 incident. The review records the timeline, root cause and contributing factors, and creates tracked corrective actions with owners and due dates.

## C-IR-03 Regulatory breach notification within 72 hours
- Owner: Chief Compliance Officer
- Frequency: Per confirmed breach
- Risks: R-REG-FINE, R-DATA-LEAK
- Systems: ledger-core, analytics-warehouse, genai-assistant
- Related: C-DP-03
- Evidence: Regulator notification record

When a personal data breach is confirmed, Compliance must notify the relevant regulator within 72 hours of becoming aware of it, unless the breach is unlikely to result in risk to individuals. Affected customers are informed without undue delay when the risk is high.
