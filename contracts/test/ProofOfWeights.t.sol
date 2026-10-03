// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {ProofOfWeights} from "../src/ProofOfWeights.sol";
import {MockERC20} from "../src/MockERC20.sol";

contract ProofOfWeightsTest is Test {
    MockERC20 token;
    ProofOfWeights reg;
    address validator = address(0x1111);
    address provider = address(0x2222);
    address challenger = address(0x3333);
    uint256 constant STAKE = 100 ether;
    uint256 constant BOND = 1 ether;

    function setUp() public {
        token = new MockERC20();
        reg = new ProofOfWeights(address(token), validator, STAKE, BOND);
        token.mint(provider, STAKE);
        vm.prank(provider);
        token.approve(address(reg), STAKE);
        vm.deal(challenger, 10 ether);
        vm.deal(validator, 0);
    }

    function _registerAndProof() internal returns (uint256 pid, uint256 proofId) {
        vm.prank(provider);
        reg.registerProvider(bytes32(uint256(1)));
        pid = 1;
        vm.prank(provider);
        reg.submitProof(pid, bytes32(uint256(2)), bytes32(uint256(3)), "ipfs://blob");
        proofId = 1;
    }

    function testHonestNotSlashed() public {
        (uint256 pid,) = _registerAndProof();
        assertEq(reg.providerStake(pid), STAKE);
        uint8[] memory h = reg.getTamperHistory(pid);
        assertEq(h.length, 0);
    }

    function testTamperedSlashedTypedClass() public {
        (uint256 pid, uint256 proofId) = _registerAndProof();
        vm.prank(challenger);
        reg.challenge{value: BOND}(pid, proofId);
        uint256 vBefore = validator.balance;
        vm.prank(validator);
        reg.resolveChallenge(1, true, 2); // quant_4bit
        assertEq(reg.providerStake(pid), 0);
        uint8[] memory h = reg.getTamperHistory(pid);
        assertEq(h.length, 1);
        assertEq(h[0], 2);
        assertEq(token.balanceOf(challenger), STAKE / 2);
        assertEq(token.balanceOf(validator), STAKE - STAKE / 2);
        assertEq(challenger.balance, 10 ether);
        assertEq(validator.balance, vBefore);
    }

    function testBinaryTamperedClass() public {
        (uint256 pid, uint256 proofId) = _registerAndProof();
        vm.prank(challenger);
        reg.challenge{value: BOND}(pid, proofId);
        vm.prank(validator);
        reg.resolveChallenge(1, true, 255);
        uint8[] memory h = reg.getTamperHistory(pid);
        assertEq(h[0], 255);
    }

    function testFalseChallengerLosesBond() public {
        (uint256 pid, uint256 proofId) = _registerAndProof();
        uint256 cBefore = challenger.balance;
        vm.prank(challenger);
        reg.challenge{value: BOND}(pid, proofId);
        vm.prank(validator);
        reg.resolveChallenge(1, false, 0);
        assertEq(reg.providerStake(pid), STAKE);
        assertEq(challenger.balance, cBefore - BOND);
        assertEq(validator.balance, BOND);
        assertEq(reg.getTamperHistory(pid).length, 0);
    }

    function testChallengeOutsideWindowReverts() public {
        (uint256 pid, uint256 proofId) = _registerAndProof();
        vm.warp(block.timestamp + 2 hours + 1);
        vm.prank(challenger);
        vm.expectRevert(bytes("window"));
        reg.challenge{value: BOND}(pid, proofId);
    }

    function testDoubleSlashReverts() public {
        (uint256 pid, uint256 proofId) = _registerAndProof();
        vm.prank(provider);
        reg.submitProof(pid, bytes32(uint256(9)), bytes32(uint256(8)), "ipfs://b2");
        uint256 proof2 = 2;
        vm.prank(challenger);
        reg.challenge{value: BOND}(pid, proofId);
        vm.prank(validator);
        reg.resolveChallenge(1, true, 4);
        vm.deal(challenger, 10 ether);
        vm.prank(challenger);
        reg.challenge{value: BOND}(pid, proof2);
        vm.prank(validator);
        vm.expectRevert(bytes("double-slash"));
        reg.resolveChallenge(2, true, 1);
    }

    function testOnlyValidatorResolves() public {
        (uint256 pid, uint256 proofId) = _registerAndProof();
        vm.prank(challenger);
        reg.challenge{value: BOND}(pid, proofId);
        vm.prank(challenger);
        vm.expectRevert(bytes("validator"));
        reg.resolveChallenge(1, true, 1);
    }
}
