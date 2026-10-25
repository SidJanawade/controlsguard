# AI Governance Policy

This policy governs generative AI applications that use hosted or open foundation models. It does not cover training or fine-tuning of models.

## C-AI-01 Human review of AI-generated customer-facing output
- Owner: Head of AI Governance
- Frequency: Per output
- Risks: R-AI-HARM
- Systems: genai-assistant
- Related: C-AI-04
- Evidence: Reviewer approval log

Any AI-generated content that reaches a customer must be reviewed and approved by a trained human before release. Internal-only assistants must display a notice that output may be inaccurate and must cite the sources used.

## C-AI-02 Prompt and model change approval with evaluation evidence
- Owner: Head of AI Governance
- Frequency: Per change
- Risks: R-AI-HARM, R-CHANGE-FAIL
- Systems: genai-assistant
- Related: C-CM-01
- Evidence: Evaluation report attached to the change ticket

A change to a system prompt, retrieval configuration or model version is a production change. It requires an evaluation report showing groundedness, refusal behaviour and regression results against the approved test set before approval.

## C-AI-03 Prompt and response logging with PII redaction
- Owner: Head of AI Governance
- Frequency: Continuous
- Risks: R-DATA-LEAK, R-REG-FINE
- Systems: genai-assistant
- Related: C-TP-03, C-DP-01
- Evidence: Audit log sample showing redaction

Every prompt and response is logged for audit with a trace identifier. Personal data such as email addresses, phone numbers, government identifiers and card numbers must be redacted before logs are written, and logs are retained according to the retention schedule.

## C-AI-04 Prompt-injection and guardrail testing before release
- Owner: Head of AI Governance
- Frequency: Per release
- Risks: R-AI-HARM, R-DATA-LEAK
- Systems: genai-assistant
- Related: C-AI-01, C-AI-02
- Evidence: Red-team test results

Before each release the application is tested against direct and indirect prompt-injection attempts, including instructions hidden in retrieved documents. The release is blocked if any test leaks the system prompt, bypasses a guardrail, or calls a tool outside its allow-list.
