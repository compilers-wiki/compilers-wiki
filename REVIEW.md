# Review Policy

## When should I request a review?

- functional changes, new documentation, new description

If a function is very simple and has no functional modifications, possibly in this case, requesting for a review is a waste of other people's time, then you don't need a review. Typically include the following scenarios:

- typo fixes
- trim trailing whitespace

## Accept a PR

A PR only needs one "approval" to be submitted to the mainline. Anyone can express themselves accepting a PR, but usually we expect good judgment from the person accepting the PR.

## How to land a PR

Usually, after a PR is accepted, it needs to be rebased to the latest `main` branch, and then the redundant commits are squashed into one. Make a "squash commit" on the mainline. We usually use the command line to accomplish this task.
