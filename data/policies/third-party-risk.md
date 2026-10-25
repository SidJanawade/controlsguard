# Third-Party Risk Management Policy

This policy governs how external vendors, including technology and AI providers, are onboarded and monitored.

## C-TP-01 Vendor due diligence before onboarding
- Owner: Head of Vendor Risk
- Frequency: Per vendor
- Risks: R-VENDOR
- Systems: payments-gateway
- Related: C-TP-02
- Evidence: Completed due diligence questionnaire and risk rating

No vendor may process company data until due diligence is complete. Due diligence covers security posture, financial stability, business continuity and subcontractor use, and ends with a risk rating approved by Vendor Risk.

## C-TP-02 Annual vendor reassessment
- Owner: Head of Vendor Risk
- Frequency: Annual
- Risks: R-VENDOR
- Systems: payments-gateway
- Related: C-TP-01
- Evidence: Reassessment report

High-risk vendors are reassessed every year, and medium-risk vendors every two years. The reassessment re-checks certifications, incident history and service level performance, and any open finding must have a remediation date.

## C-TP-03 AI vendor data-use terms review
- Owner: Head of Vendor Risk
- Frequency: Per AI vendor and per contract renewal
- Risks: R-VENDOR, R-DATA-LEAK
- Systems: genai-assistant
- Related: C-AI-03
- Evidence: Legal review of data-use and retention terms

Before a hosted foundation model is used, Legal and Vendor Risk review the provider terms. The review must confirm that prompts and outputs are not used to train the provider models, that retention is limited, and that data residency requirements are met.
