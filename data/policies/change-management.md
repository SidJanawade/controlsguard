# Change Management Policy

This policy governs how changes reach production. It applies to application code, infrastructure configuration and model or prompt changes.

## C-CM-01 Peer review and approval before production deployment
- Owner: Head of Engineering Productivity
- Frequency: Per change
- Risks: R-CHANGE-FAIL
- Systems: ledger-core, payments-gateway, prod-servers
- Related: C-CM-02, C-CM-03
- Evidence: Approved pull request and change ticket

Every production change requires a peer-reviewed pull request and an approved change ticket before deployment. The reviewer must be a different person from the author. Automated pipelines block deployment when the approval is missing.

## C-CM-02 Emergency change retro-approval within 2 business days
- Owner: Head of Service Delivery
- Frequency: Per emergency change
- Risks: R-CHANGE-FAIL, R-OUTAGE
- Systems: prod-servers, payments-gateway
- Related: C-CM-01
- Evidence: Retrospective approval record

During a Sev1 incident an on-call engineer may deploy an emergency change without prior approval. The change must receive retrospective approval from the change advisory board within 2 business days, and any unapproved emergency change is reported to the risk committee.

## C-CM-03 Segregation of duties between developer and deployer
- Owner: Head of Engineering Productivity
- Frequency: Per change
- Risks: R-INSIDER, R-CHANGE-FAIL
- Systems: ledger-core, prod-servers
- Related: C-AC-02, C-CM-01
- Evidence: Pipeline role configuration export

The person who writes a change must not be the person who promotes it to production. Deployment rights sit with the release pipeline service account, and developers hold no standing write access to production.
