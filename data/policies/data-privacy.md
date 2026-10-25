# Data Privacy and Protection Policy

This policy covers personal data and confidential business data across production and non-production environments.

## C-DP-01 PII classification and masking in non-production
- Owner: Chief Data Office
- Frequency: Continuous
- Risks: R-DATA-LEAK
- Systems: trade-db, analytics-warehouse
- Related: C-DP-02
- Evidence: Data classification catalogue and masking job logs

All columns holding personally identifiable information must be classified in the data catalogue. Data copied to development, test or analytics environments must be masked or tokenised first. Unmasked production extracts in non-production environments are a policy breach.

## C-DP-02 Encryption at rest and in transit
- Owner: Head of Infrastructure Security
- Frequency: Continuous
- Risks: R-DATA-LEAK
- Systems: trade-db, ledger-core, payments-gateway, analytics-warehouse
- Related: C-DP-01
- Evidence: Key management configuration and TLS scan results

Confidential data must be encrypted at rest with managed keys that rotate every 12 months, and encrypted in transit using TLS 1.2 or higher. Weekly scans flag endpoints that accept older protocol versions.

## C-DP-03 Data retention and deletion schedule
- Owner: Chief Data Office
- Frequency: Monthly
- Risks: R-REG-FINE
- Systems: analytics-warehouse, ledger-core
- Related: C-IR-03
- Evidence: Deletion job reports

Records are kept only for the retention period defined in the retention schedule, typically 7 years for financial records. A monthly job deletes expired records and produces a report. Legal holds suspend deletion for the affected records.
