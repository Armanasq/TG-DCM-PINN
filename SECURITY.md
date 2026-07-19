# Security policy

## Supported version

Security fixes are applied to the latest release on the default branch.

## Reporting

Report vulnerabilities privately through GitHub security advisories for
`GinkgoQ/TG-DCM-PINN`. Do not disclose exploitable details in a public issue.

## Model and data safety

- The released checkpoint is loaded with `torch.load(..., weights_only=True)`.
- Treat checkpoints from other sources as untrusted files.
- Dataset preparation rejects ZIP path traversal and symbolic-link entries.
- Raw datasets are downloaded from third-party sources and remain governed by
  their original licenses and integrity guarantees.
