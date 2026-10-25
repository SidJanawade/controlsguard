# Risk Register

The risk register lists the technology risks that controls are designed to mitigate and the systems each risk affects.

## R-UNAUTH Unauthorized access
- Category: Security
- Affects: ledger-core, trade-db, payments-gateway

A user or attacker gains access to systems or data beyond what their role requires.

## R-INSIDER Insider misuse
- Category: Conduct
- Affects: trade-db, prod-servers, ledger-core

An employee or contractor abuses legitimate access to alter, steal or conceal data or transactions.

## R-CHANGE-FAIL Failed or unauthorized change
- Category: Operational
- Affects: ledger-core, payments-gateway, prod-servers, genai-assistant

A change introduces a defect or is made without proper approval, causing incorrect processing or instability.

## R-OUTAGE Service outage
- Category: Operational
- Affects: payments-gateway, ledger-core, trade-db

A system becomes unavailable or degraded for longer than the agreed tolerance.

## R-DATA-LEAK Data leakage
- Category: Security
- Affects: trade-db, ledger-core, analytics-warehouse, genai-assistant

Confidential or personal data is exposed to people or systems that are not authorised to hold it.

## R-VENDOR Third-party failure
- Category: Third party
- Affects: payments-gateway, ledger-core, genai-assistant

A vendor fails, is breached or changes terms in a way that harms the company or its customers.

## R-REG-FINE Regulatory non-compliance
- Category: Compliance
- Affects: ledger-core, analytics-warehouse, genai-assistant

The company breaches a legal or regulatory obligation and is exposed to fines or sanctions.

## R-AI-HARM Harmful or ungrounded AI output
- Category: AI
- Affects: genai-assistant

A generative AI application produces inaccurate, unsafe or unsupported output that someone relies on.
