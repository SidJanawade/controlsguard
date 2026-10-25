# Access Control Policy

This policy governs who may access production systems and data. All controls apply to systems classified as in scope in the system inventory.

## C-AC-01 Quarterly access recertification
- Owner: Head of Identity and Access Management
- Frequency: Quarterly
- Risks: R-UNAUTH
- Systems: ledger-core, trade-db, payments-gateway
- Related: C-AC-03
- Evidence: Signed recertification report with reviewer attestation

Line managers must review every user entitlement on in-scope systems each quarter and attest that the access is still required. Entitlements that are not attested within 10 business days are revoked automatically. The signed recertification report is retained for audit.

## C-AC-02 Privileged access management with session recording
- Owner: Head of Infrastructure Security
- Frequency: Continuous
- Risks: R-UNAUTH, R-INSIDER
- Systems: prod-servers, trade-db
- Related: C-CM-03
- Evidence: Session recordings and PAM checkout logs

Administrator and root credentials are held in a privileged access management vault. Engineers check out credentials for a time-boxed session of at most 4 hours, and every privileged session is recorded. Shared administrator passwords are prohibited.

## C-AC-03 Joiner-mover-leaver access removal within 24 hours
- Owner: Head of Identity and Access Management
- Frequency: Event-driven
- Risks: R-UNAUTH
- Systems: ledger-core, trade-db, payments-gateway, analytics-warehouse
- Related: C-AC-01
- Evidence: HR feed reconciliation report

When an employee leaves or changes role, all access that is no longer required must be removed within 24 hours of the HR event. The identity platform reconciles the HR feed against system accounts daily and raises exceptions for orphaned accounts.
