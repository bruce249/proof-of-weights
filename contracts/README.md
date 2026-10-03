# Contracts

`ProofOfWeights.sol` is an optimistic registry. Verification is off-chain. One validator wallet resolves challenges.

Install Foundry (not on this machine yet), then:

```bash
cd contracts
forge install foundry-rs/forge-std --no-commit
forge test
```

Deploy to Base Sepolia only. Write address + ABI to `deployment.json`.
