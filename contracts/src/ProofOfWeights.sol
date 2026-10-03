// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20Minimal {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
}

/// @notice Optimistic registry. Verification is off-chain. One trusted validator.
contract ProofOfWeights {
    uint256 public constant DISPUTE_WINDOW = 2 hours;

    IERC20Minimal public immutable stakeToken;
    address public immutable validator;
    uint256 public immutable requiredStake;
    uint256 public immutable challengeBond;

    uint256 public nextProviderId = 1;
    uint256 public nextProofId = 1;
    uint256 public nextChallengeId = 1;

    struct Provider {
        address wallet;
        bytes32 modelHash;
        uint256 stake;
        bool slashed;
        uint8[] tamperHistory;
    }

    struct ProofRec {
        uint256 providerId;
        address submitter;
        bytes32 promptHash;
        bytes32 proofHash;
        string blobURI;
        uint256 submittedAt;
        bool challenged;
        bool slashed;
    }

    struct ChallengeRec {
        uint256 providerId;
        uint256 proofId;
        address challenger;
        uint256 bond;
        bool resolved;
    }

    mapping(uint256 => Provider) private _providers;
    mapping(address => uint256) public providerIdOf;
    mapping(uint256 => ProofRec) public proofs;
    mapping(uint256 => ChallengeRec) public challenges;

    constructor(address stakeToken_, address validator_, uint256 requiredStake_, uint256 challengeBond_) {
        require(validator_ != address(0), "validator");
        stakeToken = IERC20Minimal(stakeToken_);
        validator = validator_;
        requiredStake = requiredStake_;
        challengeBond = challengeBond_;
    }

    function registerProvider(bytes32 modelHash) external {
        require(providerIdOf[msg.sender] == 0, "registered");
        require(stakeToken.transferFrom(msg.sender, address(this), requiredStake), "stake");
        uint256 id = nextProviderId++;
        Provider storage p = _providers[id];
        p.wallet = msg.sender;
        p.modelHash = modelHash;
        p.stake = requiredStake;
        providerIdOf[msg.sender] = id;
    }

    function submitProof(uint256 providerId, bytes32 promptHash, bytes32 proofHash, string calldata blobURI)
        external
    {
        require(providerIdOf[msg.sender] == providerId, "not provider");
        require(!_providers[providerId].slashed, "slashed");
        uint256 id = nextProofId++;
        proofs[id] = ProofRec({
            providerId: providerId,
            submitter: msg.sender,
            promptHash: promptHash,
            proofHash: proofHash,
            blobURI: blobURI,
            submittedAt: block.timestamp,
            challenged: false,
            slashed: false
        });
    }

    function challenge(uint256 providerId, uint256 proofId) external payable {
        require(msg.value == challengeBond, "bond");
        ProofRec storage pr = proofs[proofId];
        require(pr.providerId == providerId, "proof");
        require(pr.submittedAt != 0, "missing");
        require(!pr.challenged, "already");
        require(block.timestamp < pr.submittedAt + DISPUTE_WINDOW, "window");
        pr.challenged = true;
        uint256 cid = nextChallengeId++;
        challenges[cid] = ChallengeRec({
            providerId: providerId,
            proofId: proofId,
            challenger: msg.sender,
            bond: msg.value,
            resolved: false
        });
    }

    function resolveChallenge(uint256 challengeId, bool upheld, uint8 tamperClass) external {
        require(msg.sender == validator, "validator");
        ChallengeRec storage ch = challenges[challengeId];
        require(ch.challenger != address(0), "missing");
        require(!ch.resolved, "resolved");
        ch.resolved = true;
        Provider storage p = _providers[ch.providerId];
        ProofRec storage pr = proofs[ch.proofId];

        if (upheld) {
            require(!p.slashed, "double-slash");
            p.slashed = true;
            p.stake = 0;
            pr.slashed = true;
            p.tamperHistory.push(tamperClass);
            uint256 half = requiredStake / 2;
            require(stakeToken.transfer(ch.challenger, half), "pay c");
            require(stakeToken.transfer(validator, requiredStake - half), "pay v");
            (bool ok, ) = payable(ch.challenger).call{value: ch.bond}("");
            require(ok, "bond back");
        } else {
            (bool ok, ) = payable(validator).call{value: ch.bond}("");
            require(ok, "bond forfeit");
        }
    }

    function getTamperHistory(uint256 providerId) external view returns (uint8[] memory) {
        return _providers[providerId].tamperHistory;
    }

    function providerStake(uint256 providerId) external view returns (uint256) {
        return _providers[providerId].stake;
    }
}
