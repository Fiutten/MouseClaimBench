# Submission baseline and ClaimBench v2 separation

Date: 2026-07-30

## Purpose

The manuscript already submitted must remain traceable. ClaimBench v2 is a
post-submission research-hardening branch, not a silent rewrite of the submitted
evidence. The goal is to prepare stronger reviewer responses and a possible
follow-up paper without losing the exact state of the first submission.

## Stable baseline

- Stable branch at start of this work: `main`.
- Stable commit before v2 work: `ff0aca9`.
- Meaning: first submitted-paper support plus initial claim-aware artifacts.

## v2 branch

- Working branch: `claimbench-v2-reviewer-hardening`.
- Scope:
  - broader adversarial benchmark;
  - threshold sensitivity over claim gates;
  - external synthetic causal validation;
  - reviewer attack suite v2;
  - release check v2.

## Non-goals

- Do not rewrite the submitted manuscript evidence silently.
- Do not change MICRONS, Sensorium, Allen, or publication-freeze numerical
  artifacts unless explicitly regenerating a new labelled release.
- Do not promote predictive results into causal, mechanistic, whole-brain, or
  complete digital-twin claims.

## Reviewer use

If reviewers ask for stronger evidence, v2 artifacts can be cited as additional
analysis only after checking that they are reproducible from a clean commit and
that their scope is described as post-submission extension or revision material.
