/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <set>
#include <thread>
#include <unordered_map>
#include <iostream>
#include <iomanip>

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

namespace grid {

class Node {
 public:
  Node(int rank, int totalNumElems, const std::shared_ptr<Context>& context) : rank_(rank) {
    int groups = context->groups;
    int groupSize = context->size / groups;
    group_ = rank / groupSize;
    groupRank_ = rank - group_ * groupSize;

    for (int i = 0; i < groupSize; i++) {
      if (i != groupRank_) {
        groupPeerRanks_.push_back(group_ * groupSize + i);
      }
    }
    leftPeerRank_ = (groupSize + groupRank_ - 1) % groupSize + group_ * groupSize;
    rightPeerRank_ = (groupRank_ + 1) % groupSize + group_ * groupSize;

    for (int i = 0; i < groups; i++) {
      if (i != group_) {
        crossGroupPeerRanks_[i] = groupRank_ + groupSize * i;
      }
    }
    sendBackupPeerRank_ = crossGroupPeerRanks_[(group_ + 1) % groups];
    recvBackupPeerRank_ = crossGroupPeerRanks_[(group_ + groups - 1) % groups];

    groupReduceNumElems_ = (totalNumElems + groups - 1) / groups;
    groupReduceOffset_ = group_ * groupReduceNumElems_;
    if (groupReduceOffset_ + groupReduceNumElems_ > totalNumElems) {
      groupReduceNumElems_ = totalNumElems - groupReduceOffset_;
    }

    // TODO: adjust minSize back to 256
    constexpr int minSize = 1;
    chunks_ = groupSize * 2;
    chunkSize_ = std::max(minSize, (groupReduceNumElems_ + chunks_ - 1) / chunks_);

  }

  std::tuple<int, int, int> getChunkPosPerRound(int round) {
    auto chunkOffset = ((2 * groupRank_) - (round & ~0x1) +
        (round & 0x1) + chunks_) % chunks_;
    auto offset = chunkOffset * chunkSize_;
    auto length = chunkSize_;
    if (offset + length <= groupReduceNumElems_) {
      // Chunk completely in range, copy full chunk.
    } else if (offset < groupReduceNumElems_) {
      // Chunk partially in range, copy partial chunk.
      length = groupReduceNumElems_ - offset;
    } else {
      // Chunk out of range, copy nothing.
      length = 0;
    }

    offset += groupReduceOffset_;
    return std::make_tuple(chunkOffset, offset, length);
  }


  int rank_;
  int group_;
  int groupRank_;
  std::vector<int> groupPeerRanks_;
  int leftPeerRank_;
  int rightPeerRank_;
  std::unordered_map<int, int> crossGroupPeerRanks_;
  int sendBackupPeerRank_;
  int recvBackupPeerRank_;

  int groupReduceOffset_;
  int groupReduceNumElems_;

  int chunks_;
  int chunkSize_;
};

}

template <typename T>
class AllreduceGridFT2 : public Algorithm {
public:
  AllreduceGridFT2(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum)
      : Algorithm(context),
        myRank_(context_->rank),
        groups_(context->groups),
        ptrs_(ptrs),
        totalNumElems_(count),
        bytes_(totalNumElems_ * sizeof(T)),
        fn_(fn){

    for (int rank = 0; rank < contextSize_; ++rank) {
      allNodes_.emplace_back(rank, totalNumElems_, context_);
    }

    grid::Node& myNode = allNodes_[myRank_];
    chunks_ = myNode.chunks_;
    slotOffset_ = this->context_->nextSlot(3 * contextSize_ * contextSize_);

    // Setup in-group ring buffers
    auto& leftPair = this->getPair(myNode.leftPeerRank_);
    auto& rightPair = this->getPair(myNode.rightPeerRank_);

    auto leftPairSlot = getSlot(myRank_, myNode.leftPeerRank_);
    auto rightPairSlot = getSlot(myRank_, myNode.rightPeerRank_);

    for (int i = 0; i < 2; i++) {
      int recvSize = myNode.chunkSize_;
      ringInbox_.emplace_back(recvSize);
      recvRingDataBufs_.push_back(
          leftPair->createRecvBuffer(leftPairSlot + i, &ringInbox_[i][0], recvSize * sizeof(T)));
      sendRingDataBufs_.push_back(
          rightPair->createSendBuffer(rightPairSlot + i, ptrs_[0], bytes_));
    }
    sendRingNotificationBuf_ = leftPair->createSendBuffer(leftPairSlot + 2, &dummy_, sizeof(dummy_));
    recvRingNotificationBuf_ = rightPair->createRecvBuffer(rightPairSlot + 2, &dummy_, sizeof(dummy_));

    // Setup cross-group buffers
    for (int group = 0; group < groups_; group++) {
      if (group != myNode.group_) {
        int peerRank = myNode.crossGroupPeerRanks_[group];
        int recvSize = std::max(myNode.groupReduceNumElems_, allNodes_[peerRank].groupReduceNumElems_);
        auto& pair = this->getPair(peerRank);
        auto slot = getSlot(myRank_, peerRank);

        std::vector<T> inbox(recvSize);
        crossGroupInbox_[peerRank] = inbox;
        sendCrossGroupDataBufs_[peerRank] = pair->createSendBuffer(slot, ptrs_[0], bytes_);
        recvCrossGroupDataBufs_[peerRank] =
            pair->createRecvBuffer(slot, &crossGroupInbox_[peerRank][0], recvSize * sizeof(T));
        sendCrossGroupNotificationBufs_[peerRank] = pair->createSendBuffer(slot + 2, &dummy_, sizeof(dummy_));
        recvCrossGroupNotificationBufs_[peerRank] = pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));


        // Setup cross-group backup buffers
        if (peerRank == myNode.recvBackupPeerRank_) {
          backupInbox_.reserve(recvSize);
          recvBackupDataBuf_ = pair->createRecvBuffer(slot + 1, &backupInbox_[0], recvSize * sizeof(T));
        }
        if (peerRank == myNode.sendBackupPeerRank_) {
          sendBackupDataBuf_ = pair->createSendBuffer(slot + 1, ptrs_[0], bytes_);
        }
      }
    }

    recoverySlotOffset_ = this->context_->nextSlot(contextSize_);
    recoveryNotificationBuf_ = context_->createUnboundBuffer(&recvMsg_, sizeof(recvMsg_));
    recoveryThread = std::thread(&AllreduceGridFT2::recoveryFunction, this);

  }

  virtual ~AllreduceGridFT2() {
  }

  void run() {
    // Reduce specified pointers into ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      fn_->call(ptrs_[0], ptrs_[i], totalNumElems_);
    }

    if (this->contextSize_ == 1) {
      // Broadcast ptrs_[0]
      for (int i = 1; i < ptrs_.size(); i++) {
        memcpy(ptrs_[i], ptrs_[0], bytes_);
      }
      return;
    }
    printAddr();
    printElems(&ptrs_[0][0], totalNumElems_);

    crossGroupReduceScatter();
    printElems(&ptrs_[0][0], totalNumElems_);

    inGroupReduceScatter();

    inGroupAllGather();

    crossGroupAllGather();


    if (myRank_ == getNextAvailableRank(0)) {
      signalNodeFailure(-1);
    }

    recoveryThread.join();

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }


protected:
  void crossGroupReduceScatter() {
    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      int offset = allNodes_[peerRank].groupReduceOffset_;
      int length = allNodes_[peerRank].groupReduceNumElems_;
      sendCrossGroupDataBufs_[peerRank]->send(offset * sizeof(T), length * sizeof(T));
    }

    {
      int offset = allNodes_[myRank_].groupReduceOffset_;
      int length = allNodes_[myRank_].groupReduceNumElems_;
      sendBackupDataBuf_->send(offset * sizeof(T), length * sizeof(T));
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      recvCrossGroupDataBufs_[peerRank]->waitRecv();
    }

    {
      recvBackupDataBuf_->waitRecv();
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      sendCrossGroupNotificationBufs_[peerRank]->send();
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      recvCrossGroupNotificationBufs_[peerRank]->waitRecv();
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      int offset = allNodes_[myRank_].groupReduceOffset_;
      int length = allNodes_[myRank_].groupReduceNumElems_;
      fn_->call(&ptrs_[0][offset], &crossGroupInbox_[peerRank][0], length);
    }

    {
      int recvBackupPeerRank = allNodes_[myRank_].recvBackupPeerRank_;
      int offset = allNodes_[recvBackupPeerRank].groupReduceOffset_;
      int length = allNodes_[recvBackupPeerRank].groupReduceNumElems_;
      fn_->call(&ptrs_[0][offset], &backupInbox_[0], length);
    }

  }

  void inGroupReduceScatter() {

    {
      int chunkOffset, offset, length;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, 0);
      sendRingDataBufs_[chunkOffset & 1]->send(offset * sizeof(T), length * sizeof(T));
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, 1);
      sendRingDataBufs_[chunkOffset & 1]->send(offset * sizeof(T), length * sizeof(T));
    }

    for (round_ = 2; round_ < chunks_; round_++) {
      insertFailure();

      int chunkOffset, offset, length;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, round_);

      // Wait for inbox write to complete
      recvRingDataBufs_[chunkOffset & 1]->waitRecv();

      // Reduce
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], &ringInbox_[chunkOffset & 1][0], length);
      }

      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendRingNotificationBuf_->send();

      if (proxyNodes_.empty()) {
        // Wait for notification from node on the right
        // to be sure this node can start an inbox write.
        recvRingNotificationBuf_->waitRecv();

        // Copy accumulated chunk
        sendRingDataBufs_[chunkOffset & 1]->send(offset * sizeof(T), length * sizeof(T));

      } else {
        proxyInGroupReduceScatter();
      }

      printElems(&ptrs_[0][0], totalNumElems_);
    }

  }

  void inGroupAllGather() {

    for (round_ = 0; round_ < (chunks_ - 2); round_++) {
      int chunkOffset, offset, length;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, round_);

      // Wait for inbox write to complete
      recvRingDataBufs_[chunkOffset & 1]->waitRecv();

      // Copy
      if (length > 0) {
        memcpy(&ptrs_[0][offset], &ringInbox_[chunkOffset & 1][0], length * sizeof(T));
      }

      // Skip copying in the last two rounds
      if (round_ < (chunks_ - 4)) {
        // Send notification to node on the left that
        // this node is ready for an inbox write.
        sendRingNotificationBuf_->send();

        if (proxyNodes_.empty()) {
          // Wait for notification from node on the right
          // to be sure this node can start an inbox write.
          recvRingNotificationBuf_->waitRecv();

          // Copy accumulated chunks
          sendRingDataBufs_[chunkOffset & 1]->send(offset * sizeof(T), length * sizeof(T));
        }
      }

      if (!proxyNodes_.empty()) {
        proxyInGroupAllGather();
      }

      printElems(&ptrs_[0][0], totalNumElems_);
    }



  }

  void crossGroupAllGather() {
    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      if (allNodes_[peerRank].groupRank_ == allNodes_[myRank_].groupRank_) {
        int offset = allNodes_[myRank_].groupReduceOffset_;
        int length = allNodes_[myRank_].groupReduceNumElems_;
        sendCrossGroupDataBufs_[peerRank]->send(offset * sizeof(T), length * sizeof(T));
      }
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      int offset = allNodes_[peerRank].groupReduceOffset_;
      int length = allNodes_[peerRank].groupReduceNumElems_;
      recvCrossGroupDataBufs_[peerRank]->waitRecv();
      if (length > 0) {
        memcpy(&ptrs_[0][offset], &crossGroupInbox_[peerRank][0], length * sizeof(T));
      }
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      sendCrossGroupNotificationBufs_[peerRank]->send();
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      recvCrossGroupNotificationBufs_[peerRank]->waitRecv();
    }

    if (!proxyNodes_.empty()) {
      proxyCrossGroupAllGather();
    }

    printElems(&ptrs_[0][0], totalNumElems_);

    // Final barrier to make sure every node has finished
    // Otherwise, a second all reduce call might interfere
    // with one that it still in progress on some nodes.
    sendRingNotificationBuf_->send();

    if (proxyNodes_.empty()) {
      recvRingNotificationBuf_->waitRecv();
    } else {
      proxyNodes_[0].recvRingNotificationBuf_->waitRecv();
    }

  }


  void recoveryFunction() {
    std::vector<int> allRanks(contextSize_);
    for (int rank = 0; rank < contextSize_; ++rank) {
      if (rank != myRank_) {
        allRanks.push_back(rank);
      }
    }
    while (recvMsg_ != -1) {
      recoveryNotificationBuf_->recv(allRanks, recoverySlotOffset_ + myRank_);
      recoveryNotificationBuf_->waitRecv();
      std::cout << "Unbounded message: " << recvMsg_ << std::endl;
      if (recvMsg_ >= 0) {
        failedNodeRanks_.insert(recvMsg_);
      }
    }
    int lowestAvailableRank = getNextAvailableRank(0);
    if (myRank_ == getNextAvailableRank(lowestAvailableRank + 1)) {
      recoveryNotificationBuf_->send(lowestAvailableRank, recoverySlotOffset_ + lowestAvailableRank);
      recoveryNotificationBuf_->waitSend();
    }
    std::cout << "Recovery finished" << std::endl;
  }

  void signalNodeFailure(int rank) {
    failedNodeRanks_.insert(rank);
    sendMsg_ = rank;
    std::vector<std::unique_ptr<transport::UnboundBuffer>> notifs;
    for (int dstRank : getAvailablePeerRanks()) {
        notifs.push_back(context_->createUnboundBuffer(&sendMsg_, sizeof(sendMsg_)));
        notifs[notifs.size()-1]->send(dstRank, recoverySlotOffset_ + dstRank);
    }
    for (auto & notif : notifs) {
      notif->waitSend();
    }
  }

  void repairRightPeer(int failedNodeRank) {
    std::cout << "repair" << std::endl;

    grid::Node& myNode = allNodes_[myRank_];
    grid::Node& failedNode = allNodes_[failedNodeRank];

    struct {
      int recoveryPeerRank;
      bool isRequested;
    } backupRequestMsg;

    backupRequestMsg.recoveryPeerRank = myRank_;
    backupRequestMsg.isRequested = true;

    std::vector<std::unique_ptr<transport::UnboundBuffer>> notifs;
    for (int dstRank : getCrossGroupPeers(failedNodeRank)) {
      notifs.push_back(context_->createUnboundBuffer(&backupRequestMsg, sizeof(backupRequestMsg)));
      notifs[notifs.size()-1]->send(dstRank, 9999 + dstRank);
    }
    for (auto & notif : notifs) {
      notif->waitSend();
    }

    proxyNodes_.emplace_back();

    ProxyNode& proxy = proxyNodes_.back();

    setupProxy(proxy, failedNode);

    notifs.clear();
    for (int dstRank : getCrossGroupPeers(failedNodeRank)) {
      notifs.push_back(context_->createUnboundBuffer(&backupRequestMsg, sizeof(backupRequestMsg)));
      notifs[notifs.size()-1]->send(dstRank, 9999 + dstRank);
    }
    for (auto & notif : notifs) {
      notif->waitSend();
    }

    std::cout << "finished" << std::endl;

  }

  void repairLeftPeer(int failedNodeRank) {
    std::cout << "repair" << std::endl;

    grid::Node& myNode = allNodes_[myRank_];
    grid::Node& failedNode = allNodes_[failedNodeRank];

    auto& leftPair = this->getPair(failedNode.leftPeerRank_);
    auto leftPairSlot = getSlot(myRank_, failedNode.leftPeerRank_, true);

    recvRingDataBufs_.clear();
    for (int i = 0; i < 2; i++) {
      int recvSize = myNode.chunkSize_;
      recvRingDataBufs_.push_back(
          leftPair->createRecvBuffer(leftPairSlot + i, &ringInbox_[i][0], recvSize * sizeof(T)));
    }
    sendRingNotificationBuf_ = leftPair->createSendBuffer(leftPairSlot + 2, &dummy_, sizeof(dummy_));

    std::cout << "finish" << std::endl;
  }

  void repairCrossGroupPeer(int failedNodeRank) {
    std::cout << "repair" << std::endl;

    grid::Node& myNode = allNodes_[myRank_];
    grid::Node& failedNode = allNodes_[failedNodeRank];

    struct {
      int recoveryPeerRank;
      bool isRequested;
    } backupRequestMsg;

    std::unique_ptr<transport::UnboundBuffer> repairNotificationBuf =
        context_->createUnboundBuffer(&backupRequestMsg, sizeof(backupRequestMsg));

    repairNotificationBuf->recv(failedNode.groupPeerRanks_, 9999 + myRank_);
    repairNotificationBuf->waitRecv();

    if (backupRequestMsg.isRequested) {
      int slot = getSlot(myRank_, backupRequestMsg.recoveryPeerRank, true);
      auto& pair = this->getPair(backupRequestMsg.recoveryPeerRank);
      auto sendBackupBuf = pair->createSendBuffer(slot, ptrs_[0], bytes_);

      int offset = failedNode.groupReduceOffset_;
      int length = failedNode.groupReduceNumElems_;
      sendBackupBuf->send(offset * sizeof(T), length * sizeof(T));

      repairNotificationBuf->recv(backupRequestMsg.recoveryPeerRank, 9999 + myRank_);
      repairNotificationBuf->waitRecv();
    }


    // TODO: split into multiple buffers
    myNode.crossGroupPeerRanks_[failedNode.group_] = backupRequestMsg.recoveryPeerRank;
    int slot = getSlot(myRank_, backupRequestMsg.recoveryPeerRank);
    auto& pair = this->getPair(backupRequestMsg.recoveryPeerRank);

    int recvSize = std::max(myNode.groupReduceNumElems_, failedNode.groupReduceNumElems_);
    std::vector<T> inbox(recvSize);
    crossGroupInbox_[backupRequestMsg.recoveryPeerRank] = inbox;
    recvCrossGroupDataBufs_[backupRequestMsg.recoveryPeerRank] =
        pair->createRecvBuffer(slot, &crossGroupInbox_[backupRequestMsg.recoveryPeerRank][0], recvSize * sizeof(T));
    sendCrossGroupNotificationBufs_[backupRequestMsg.recoveryPeerRank] =
        pair->createSendBuffer(slot + 2, &dummy_, sizeof(dummy_));
    recvCrossGroupNotificationBufs_[backupRequestMsg.recoveryPeerRank] =
        pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));

    std::cout << "finish" << std::endl;
  }

  std::vector<int> getCrossGroupPeers(int rank) {
    std::vector<int> peerRanks;
    for (auto& kv : allNodes_[rank].crossGroupPeerRanks_) {
      peerRanks.push_back(kv.second);
    }
    return peerRanks;
  }

  std::tuple<int, int, int> getChunkPosPerRound(int rank, int round) {
    return allNodes_[rank].getChunkPosPerRound(round);
  }

  int getNextAvailableRank(int rank) {
    for (int rank2 = rank; rank2 < contextSize_ + rank; rank2++) {
      if (failedNodeRanks_.find(rank2 % contextSize_) == failedNodeRanks_.end()) {
        return rank2 % contextSize_;
      }
    }
    return -1;
  }

  std::vector<int> getAvailablePeerRanks() {
    std::vector<int> allRanks;
    for (int rank = 0; rank < contextSize_; rank++) {
      if (failedNodeRanks_.find(rank) == failedNodeRanks_.end() && rank != myRank_) {
        allRanks.push_back(rank);
      }
    }
    return allRanks;
  }

  int getSlot(int rank1, int rank2, bool repairMode = false) {
    if (!repairMode) {
      return slotOffset_ + 3 * (std::min(rank1, rank2) * contextSize_ + std::max(rank1, rank2));
    } else {
      return slotOffset_ + 3 * (std::max(rank1, rank2) * contextSize_ + std::min(rank1, rank2));
    }
  }

  const int myRank_;

  std::vector<T*> ptrs_;
  const int totalNumElems_;
  const int bytes_;
  const ReductionFunction<T>* fn_;

  const int groups_;
  std::vector<grid::Node> allNodes_;

  std::vector<std::vector<T>> ringInbox_;
  std::unordered_map<int, std::vector<T>> crossGroupInbox_;
  std::vector<T> backupInbox_;

  int slotOffset_;
  int dummy_;

  std::vector<std::unique_ptr<transport::Buffer>> sendRingDataBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvRingDataBufs_;

  std::unique_ptr<transport::Buffer> sendRingNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvRingNotificationBuf_;

  std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendCrossGroupDataBufs_;
  std::unordered_map<int, std::unique_ptr<transport::Buffer>> recvCrossGroupDataBufs_;

  std::unique_ptr<transport::Buffer> sendBackupDataBuf_;
  std::unique_ptr<transport::Buffer> recvBackupDataBuf_;

  std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendCrossGroupNotificationBufs_;
  std::unordered_map<int, std::unique_ptr<transport::Buffer>> recvCrossGroupNotificationBufs_;

  int chunks_;
  int round_;

  std::set<int> failedNodeRanks_;

  int recoverySlotOffset_;
  int recvMsg_;
  int sendMsg_;
  std::unique_ptr<transport::UnboundBuffer> recoveryNotificationBuf_;
  std::thread recoveryThread;

  class ProxyNode {
   public:
    ProxyNode();

    int rank_;
    std::vector<T*> ptrs_;
    std::vector<std::unique_ptr<transport::Buffer>> sendRingDataBufs_;
    std::unique_ptr<transport::Buffer> recvRingNotificationBuf_;

    std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendCrossGroupDataBufs_;
    std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendCrossGroupNotificationBufs_;
    std::unordered_map<int, std::unique_ptr<transport::Buffer>> recvCrossGroupNotificationBufs_;

  };


  std::vector<ProxyNode> proxyNodes_;

  void setupProxy(ProxyNode& proxy, grid::Node& original);
  void proxyInGroupReduceScatter();
  void proxyInGroupAllGather();
  void proxyCrossGroupAllGather();


 private:
  static constexpr int wordsPerSection = 2;
  static constexpr int wordsPerLine = 3 * wordsPerSection;

  static void printBreak(T* p, int x) {
    if (0 == x % wordsPerLine) {
      std::cout << std::endl
                << &p[x] << " " << std::setfill('0') << std::setw(3) << x
                << ": ";
    } else if (0 == x % wordsPerSection) {
      std::cout << "- ";
    }
  }

  static void printElems(T* p, int count, int start = 0) {
    auto alignedStart = (start / wordsPerLine) * wordsPerLine;
    for (int x = alignedStart; x < start + count; ++x) {
      printBreak(p, x);
      if (x < start) {
        std::cout << "..... ";
      } else {
        std::cout << std::setfill('0') << std::setw(3) << p[x] << " ";
      }
    }
    std::cout << std::endl;
  }

  void printAddr() {
    for (int i = 0; i < contextSize_; i++) {
      if (i == contextRank_) continue;
      auto &pair = context_->getPair(i);
      std::cout << "Pair " << i << ": " << pair->address().str() << std::endl;
    }
  }

  void insertFailure() {
    if (round_ == 4) {
      if (myRank_ == 1) {
        sleep(5);
        return;
      }
      if (myRank_ == 0) {
        repairRightPeer(1);
      }
      if (myRank_ == 2) {
        repairLeftPeer(1);
      }
      if (myRank_ % 3 == 1) {
        repairCrossGroupPeer(1);
      }
    }

  }
};

} // namespace gloo
