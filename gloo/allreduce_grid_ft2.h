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
#include <mutex>
#include <condition_variable>
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
    if (myRank_ == 1) {
      exit(0);
    }
    for (int rank = 0; rank < contextSize_; ++rank) {
      allNodes_.emplace_back(rank, totalNumElems_, context_);
    }

    grid::Node& myNode = allNodes_[myRank_];
    chunks_ = myNode.chunks_;
    slotOffset_ = this->context_->nextSlot(4 * contextSize_ * contextSize_);

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
    msg_.reserve(contextSize_);
    for (int group = 0; group < groups_; group++) {
      if (group != myNode.group_) {
        int peerRank = myNode.crossGroupPeerRanks_[group];
        int recvSize = std::max(myNode.groupReduceNumElems_, allNodes_[peerRank].groupReduceNumElems_);
        auto& pair = this->getPair(peerRank);
        auto slot = getSlot(myRank_, peerRank);

        crossGroupInbox_[peerRank] = std::vector<T>();
        crossGroupInbox_[peerRank].reserve(recvSize);

        sendCrossGroupDataBufs_[peerRank] = pair->createSendBuffer(slot, ptrs_[0], bytes_);
        recvCrossGroupDataBufs_[peerRank] =
            pair->createRecvBuffer(slot, &crossGroupInbox_[peerRank][0], recvSize * sizeof(T));
        sendCrossGroupNotificationBufs_[peerRank] = pair->createSendBuffer(slot + 2, &dummy_, sizeof(dummy_));
        recvCrossGroupNotificationBufs_[peerRank] = pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));
        sendConfirmationBufs_[peerRank] = pair->createSendBuffer(slot + 3, &msg_[myRank_], sizeof(dummy_));
        recvConfirmationBufs_[peerRank] = pair->createRecvBuffer(slot + 3, &msg_[peerRank], sizeof(dummy_));


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
    requestSlotOffset_ = this->context_->nextSlot(contextSize_);
    recoveryNotificationBuf_ = context_->createUnboundBuffer(&recvMsg_, sizeof(recvMsg_));
    recoveryThread_ = std::thread(&AllreduceGridFT2::recoveryFunction, this);

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

    crossGroupReduceScatter();
    printElems(&ptrs_[0][0], totalNumElems_);

    inGroupReduceScatter();

    inGroupAllGather();

    crossGroupAllGather();


    if (leftPairRepairThread_.joinable()) {
      leftPairRepairThread_.join();
    }
    if (crossGroupRepairThread_.joinable()) {
      crossGroupRepairThread_.join();
    }
    for (int i = 0; i < proxyThreads_.size(); i++) {
      if (proxyThreads_[i].joinable()) {
        proxyThreads_[i].join();
      }
    }


    signalCompletion();

    recoveryThread_.join();
    context_->updateFailedNodes(failedNodeRanks_);

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }


protected:
  void crossGroupReduceScatter() {
    std::unique_lock<std::mutex> lock(crossGroupMutex_);
    phase_ = CrossGroupReduceScatter;

    auto send = [&](std::unique_ptr<transport::Buffer>& buf, int dstRank, int offset, int length) {
      std::cout << "send data " << " ("  << ptrs_[0][offset] << "...)"<< std::endl;
      bool success = buf->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) {
        if (!crossGroupRepairThread_.joinable()) {
          crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, dstRank);
        }
        crossGroupCV_.wait(lock);
      }
    };

    auto recv = [&](std::unique_ptr<transport::Buffer>& buf, int srcRank) {
      bool success = buf->tryWaitRecv();
      std::cout << "recv data " << " ("  << crossGroupInbox_[srcRank][0] << "...)"<< std::endl;
      if (!success) {
        if (!crossGroupRepairThread_.joinable()) {
          crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, srcRank);
        }
        crossGroupCV_.wait(lock);
      }
      return success;
    };

    auto confirm = [&]() {
      msg_[myRank_] = myRank_;
      for (auto peerRank : getCrossGroupPeers(myRank_)) {
        bool success = sendConfirmationBufs_[peerRank]->trySend();
        if (!success) {
          if (!crossGroupRepairThread_.joinable()) {
            crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, peerRank);
          }
          crossGroupCV_.wait(lock);
        }
      }

      for (auto peerRank : getCrossGroupPeers(myRank_)) {
        bool success = recvConfirmationBufs_[peerRank]->tryWaitRecv();
        if (!success) {
          if (!crossGroupRepairThread_.joinable()) {
            crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, peerRank);
          }
          crossGroupCV_.wait(lock);
        } else {
          if (msg_[peerRank] != peerRank) {
            std::cout << "Peer " << peerRank << " failed with " << msg_[peerRank] << std::endl;
            allNodes_[myRank_].crossGroupPeerRanks_.erase(allNodes_[peerRank].group_);
          }
        }
      }
    };

    int sendBackupPeerRank = allNodes_[myRank_].sendBackupPeerRank_;
    int recvBackupPeerRank = allNodes_[myRank_].recvBackupPeerRank_;

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      int offset = allNodes_[peerRank].groupReduceOffset_;
      int length = allNodes_[peerRank].groupReduceNumElems_;
      send(sendCrossGroupDataBufs_[peerRank], peerRank, offset, length);
    }

    if (!isNodeFailed(sendBackupPeerRank)) {
      int offset = allNodes_[myRank_].groupReduceOffset_;
      int length = allNodes_[myRank_].groupReduceNumElems_;
      send(sendBackupDataBuf_, sendBackupPeerRank, offset, length);
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      recv(recvCrossGroupDataBufs_[peerRank], peerRank);
    }

    if (!isNodeFailed(recvBackupPeerRank)) {
      recv(recvBackupDataBuf_, recvBackupPeerRank);
    }

    confirm();

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      int offset = allNodes_[myRank_].groupReduceOffset_;
      int length = allNodes_[myRank_].groupReduceNumElems_;
      fn_->call(&ptrs_[0][offset], &crossGroupInbox_[peerRank][0], length);
    }

    {
      int offset = allNodes_[recvBackupPeerRank].groupReduceOffset_;
      int length = allNodes_[recvBackupPeerRank].groupReduceNumElems_;
      fn_->call(&ptrs_[0][offset], &backupInbox_[0], length);
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      sendCrossGroupNotificationBufs_[peerRank]->trySend();
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      recvCrossGroupNotificationBufs_[peerRank]->tryWaitRecv();
    }

    lock.unlock();

  }

  void inGroupReduceScatter() {
    phase_ = InGroupReduceScatter;

    auto waitData = [&](int chunkOffset) {
      std::cout << "wait data at " << chunkOffset;
      std::unique_lock<std::mutex> lock(leftPairMutex_);
      std::cout << "(acquire lock)" << std::endl;
      requestRoundData_ = true;
      roundNotificationSent_ = false;
      bool success = recvRingDataBufs_[chunkOffset & 1]->tryWaitRecv();
      std::cout << " ("  << ringInbox_[chunkOffset & 1][0] << ")" << std::endl;
      if (!success) {
        // TODO: handle existing repair thread
//        leftPairRepairThread_ = std::thread(&AllreduceGridFT2::repairLeftPeer, this, true);
        leftPairCV_.wait(lock);
        recvRingDataBufs_[chunkOffset & 1]->waitRecv();
      }
      requestRoundData_ = false;
      lock.unlock();
    };

    auto sendNotification = [&]() {
      std::cout << "send notification" << std::endl;
      std::unique_lock<std::mutex> lock(leftPairMutex_);
      bool success = sendRingNotificationBuf_->trySend();
      if (!success) {
//        leftPairRepairThread_ = std::thread(&AllreduceGridFT2::repairLeftPeer, this, false);
        leftPairCV_.wait(lock);
        sendRingNotificationBuf_->send();
      }
      roundNotificationSent_ = true;
      lock.unlock();
    };

    auto waitNotification = [&]() {
      if (!proxyNodes_.empty())  return;
      std::cout << "wait notification" << std::endl;
      bool success = recvRingNotificationBuf_->tryWaitRecv();
      if (!success) {
        repairRightPeer(true);
      }
    };

    auto sendData = [&](int chunkOffset, int offset, int length) {
      if (!proxyNodes_.empty())  return;
      std::cout << "send data at " << chunkOffset << " ("  << ptrs_[0][offset] << ")"<< std::endl;
      bool success = sendRingDataBufs_[chunkOffset & 1]->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) {
        repairRightPeer();
      }
    };


    {
      int chunkOffset, offset, length;
      round_ = 0;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, 0);
      sendData(chunkOffset, offset, length);
      if (!proxyNodes_.empty()) {
        proxyInGroupReduceScatter();
      }
      round_ = 1;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, 1);
      sendData(chunkOffset, offset, length);
      if (!proxyNodes_.empty()) {
        proxyInGroupReduceScatter();
      }
    }

    for (round_ = 2; round_ < chunks_; round_++) {
      std::cout << "in-group reduce-scatter round " << round_ << std::endl;
      insertFailure();

      int chunkOffset, offset, length;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, round_);

      // Wait for inbox write to complete
      waitData(chunkOffset);

      // Reduce
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], &ringInbox_[chunkOffset & 1][0], length);
      }

      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendNotification();


      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
      waitNotification();

      // Copy accumulated chunk
      sendData(chunkOffset, offset, length);

      if (!proxyNodes_.empty()) {
        proxyInGroupReduceScatter();
      }

      printElems(&ptrs_[0][0], totalNumElems_);

    }

  }

  void inGroupAllGather() {
    phase_ = InGroupAllGather;

    auto waitData = [&](int chunkOffset) {
      std::cout << "wait data at " << chunkOffset << std::endl;
      std::unique_lock<std::mutex> lock(leftPairMutex_);
      bool success = recvRingDataBufs_[chunkOffset & 1]->tryWaitRecv();
      if (!success) {
        // TODO: handle existing repair thread
//        leftPairRepairThread_ = std::thread(&AllreduceGridFT2::repairLeftPeer, this, true);
        requestRoundData_ = true;
        leftPairCV_.wait(lock);
        recvRingDataBufs_[chunkOffset & 1]->waitRecv();
      }
      lock.unlock();
    };

    auto sendNotification = [&]() {
      std::cout << "send notification" << std::endl;
      std::unique_lock<std::mutex> lock(leftPairMutex_);
      bool success = sendRingNotificationBuf_->trySend();
      if (!success) {
//        leftPairRepairThread_ = std::thread(&AllreduceGridFT2::repairLeftPeer, this, false);
        requestRoundData_ = false;
        leftPairCV_.wait(lock);
        sendRingNotificationBuf_->send();
      }
      lock.unlock();
    };

    auto waitNotification = [&]() {
      if (!proxyNodes_.empty())  return;
      std::cout << "wait notification" << std::endl;
      bool success = recvRingNotificationBuf_->tryWaitRecv();
      if (!success) {
        repairRightPeer(true);
      }
    };

    auto sendData = [&](int chunkOffset, int offset, int length) {
      if (!proxyNodes_.empty())  return;
      std::cout << "send data at " << chunkOffset << std::endl;
      bool success = sendRingDataBufs_[chunkOffset & 1]->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) {
        repairRightPeer();
      }
    };

    for (round_ = 0; round_ < (chunks_ - 2); round_++) {
      std::cout << "in-group all-gather round " << round_ << std::endl;
      int chunkOffset, offset, length;
      std::tie(chunkOffset, offset, length) = getChunkPosPerRound(myRank_, round_);

      // Wait for inbox write to complete
      waitData(chunkOffset);

      // Copy
      if (length > 0) {
        memcpy(&ptrs_[0][offset], &ringInbox_[chunkOffset & 1][0], length * sizeof(T));
      }

      // Skip copying in the last two rounds
      if (round_ < (chunks_ - 4)) {
        // Send notification to node on the left that
        // this node is ready for an inbox write.
        sendNotification();

        if (proxyNodes_.empty()) {
          // Wait for notification from node on the right
          // to be sure this node can start an inbox write.
          waitNotification();

          // Copy accumulated chunks
          sendData(chunkOffset, offset, length);
        }
      }

      if (!proxyNodes_.empty()) {
        proxyInGroupAllGather();
      }

      printElems(&ptrs_[0][0], totalNumElems_);


    }

  }

  void crossGroupAllGather() {
    std::unique_lock<std::mutex> lock(crossGroupMutex_);
    phase_ = CrossGroupAllGather;

    auto send = [&](std::unique_ptr<transport::Buffer>& buf, int dstRank, int offset, int length) {
      std::cout << "send data " << " ("  << ptrs_[0][offset] << "...)"<< std::endl;
      bool success = buf->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) {
        if (!crossGroupRepairThread_.joinable()) {
          crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, dstRank);
        }
        crossGroupCV_.wait(lock);
      }
    };

    auto recv = [&](std::unique_ptr<transport::Buffer>& buf, int srcRank) {
      bool success = buf->tryWaitRecv();
      std::cout << "recv data " << " ("  << crossGroupInbox_[srcRank][0] << "...)"<< std::endl;
      if (!success) {
        if (!crossGroupRepairThread_.joinable()) {
          crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, srcRank);
        }
        crossGroupCV_.wait(lock);
      }
      return success;
    };

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      if (allNodes_[peerRank].groupRank_ == allNodes_[myRank_].groupRank_) {
        int offset = allNodes_[myRank_].groupReduceOffset_;
        int length = allNodes_[myRank_].groupReduceNumElems_;
        send(sendCrossGroupDataBufs_[peerRank], peerRank, offset, length);
      }
    }

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      int offset = allNodes_[peerRank].groupReduceOffset_;
      int length = allNodes_[peerRank].groupReduceNumElems_;
      bool success = recv(recvCrossGroupDataBufs_[peerRank], peerRank);
      if (length > 0 && success) {
        memcpy(&ptrs_[0][offset], &crossGroupInbox_[peerRank][0], length * sizeof(T));
      }
    }

    phase_ = Completed;
    proxyCV_.notify_all();

    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      sendCrossGroupNotificationBufs_[peerRank]->trySend();
    }


    for (auto peerRank : getCrossGroupPeers(myRank_)) {
      recvCrossGroupNotificationBufs_[peerRank]->waitRecv();
    }

    lock.unlock();

    if (!proxyNodes_.empty()) {
      proxyCrossGroupAllGather();
    }

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
    std::cout << "Recovery start" << std::endl;
    std::vector<int> allRanks(contextSize_);
    for (int rank = 0; rank < contextSize_; ++rank) {
      if (rank != myRank_) {
        allRanks.push_back(rank);
      }
    }
    int numNodeCompleted = 0;
    while (numNodeCompleted + failedNodeRanks_.size() != contextSize_ - 1) {
      recoveryNotificationBuf_->recv(allRanks, recoverySlotOffset_ + myRank_);
      recoveryNotificationBuf_->waitRecv();
      std::cout << "Unbounded message: " << recvMsg_ << std::endl;
      if (recvMsg_ >= 0) {
        int rank = recvMsg_;
        failedNodeRanks_.insert(rank);
        if (isCrossGroupPeers(myRank_, rank) && !crossGroupRepairThread_.joinable()) {
          crossGroupRepairThread_ = std::thread(&AllreduceGridFT2::repairCrossGroupPeer, this, rank);
        }
        if (allNodes_[myRank_].leftPeerRank_ == rank && !leftPairRepairThread_.joinable()) {
          leftPairRepairThread_ = std::thread(&AllreduceGridFT2::repairLeftPeer, this);
        }
        if (isGroupPeers(myRank_, rank)) {
          proxyThreads_.emplace_back(&AllreduceGridFT2::repairGroupPeer, this, rank);
        }
      }
      if (recvMsg_ == -1) {
        numNodeCompleted++;
      }
    }

    std::cout << "Recovery finished" << std::endl;
  }

  void signalNodeFailure(int rank) {
    failedNodeRanks_.insert(rank);
    sendMsg_ = rank;
    std::vector<std::unique_ptr<transport::UnboundBuffer>> notifs;
    for (int dstRank : getAvailablePeerRanks()) {
      notifs.push_back(context_->createUnboundBuffer(&sendMsg_, sizeof(sendMsg_)));
      notifs.back()->trySend(dstRank, recoverySlotOffset_ + dstRank);
    }
    for (auto & notif : notifs) {
      notif->tryWaitSend();
    }
  }

  void signalCompletion() {
    std::cout << "COMPLETED" << std::endl;
    sendMsg_ = -1;
    std::vector<std::unique_ptr<transport::UnboundBuffer>> notifs;
    for (int dstRank : getAvailablePeerRanks()) {
      notifs.push_back(context_->createUnboundBuffer(&sendMsg_, sizeof(sendMsg_)));
      notifs.back()->trySend(dstRank, recoverySlotOffset_ + dstRank);
    }
    for (auto & notif : notifs) {
      notif->tryWaitSend();
    }
  }

  void repairRightPeer(bool requestNotification = false);


  void repairLeftPeer();


  void repairGroupPeer(int failedNodeRank);


  void repairCrossGroupPeer(int failedNodeRank);

  std::vector<int> getCrossGroupPeers(int rank) {
    std::vector<int> peerRanks;
    for (auto& kv : allNodes_[rank].crossGroupPeerRanks_) {
      peerRanks.push_back(kv.second);
    }
    return peerRanks;
  }

  std::vector<int> getGroupPeers(int rank) {
    std::vector<int> peerRanks;
    for (int peerRank : allNodes_[rank].groupPeerRanks_) {
      if (!isNodeFailed(peerRank)) {
        peerRanks.push_back(peerRank);
      }
    }
    return peerRanks;
  }

  bool isCrossGroupPeers(int rank1, int rank2) {
    std::vector<int> rank1PeerRanks = getCrossGroupPeers(rank1);
    for (int peerRank : rank1PeerRanks) {
      if (peerRank == rank2) {
        return true;
      }
    }
    return false;
  }

  bool isGroupPeers(int rank1, int rank2) {
    std::vector<int> rank1PeerRanks = allNodes_[rank1].groupPeerRanks_;
    for (int peerRank : rank1PeerRanks) {
      if (peerRank == rank2) {
        return true;
      }
    }
    return false;
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

  bool isNodeFailed(int rank) {
    return failedNodeRanks_.find(rank) != failedNodeRanks_.end();
  }

  int getActualGroupRank(int rank) {
    int originalRank = allNodes_[rank].groupRank_;
    int actualRank = 0;
    for (int groupPeerRank : allNodes_[rank].groupPeerRanks_) {
      if (groupPeerRank > rank) {
        break;
      }
      if (!isNodeFailed(groupPeerRank)) {
        actualRank++;
      }
    }
    return actualRank;
  }

  int getActualGroupSize(int rank) {
    int groupSize = isNodeFailed(rank) ? 0 : 1;
    for (int groupPeerRank : allNodes_[rank].groupPeerRanks_) {
      if (!isNodeFailed(groupPeerRank)) {
        groupSize++;
      }
    }
    return groupSize;
  }


  int getSlot(int rank1, int rank2, bool repairMode = false) {
    if (!repairMode) {
      return slotOffset_ + 4 * (std::min(rank1, rank2) * contextSize_ + std::max(rank1, rank2));
    } else {
      return slotOffset_ + 4 * (std::max(rank1, rank2) * contextSize_ + std::min(rank1, rank2));
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

  std::vector<int> msg_;
  std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendConfirmationBufs_;
  std::unordered_map<int, std::unique_ptr<transport::Buffer>> recvConfirmationBufs_;

  int chunks_;

  enum Phase {CrossGroupReduceScatter, InGroupReduceScatter, InGroupAllGather, CrossGroupAllGather, Completed};
  Phase phase_;
  int round_;

  std::set<int> failedNodeRanks_;

  int recoverySlotOffset_;
  int requestSlotOffset_;
  int recvMsg_;
  int sendMsg_;
  std::unique_ptr<transport::UnboundBuffer> recoveryNotificationBuf_;
  std::thread recoveryThread_;

  std::thread crossGroupRepairThread_;
  std::mutex crossGroupMutex_;
  std::condition_variable crossGroupCV_;

  std::thread leftPairRepairThread_;
  std::mutex leftPairMutex_;
  std::condition_variable leftPairCV_;
  bool requestRoundData_ = true;
  bool roundNotificationSent_ = false;

  std::vector<std::thread> proxyThreads_;
  std::mutex proxyMutex_;
  std::condition_variable proxyCV_;

  class ProxyNode {
   public:
    ProxyNode();

    int rank_;
    std::vector<T*> ptrs_;

    int startRound_;
    Phase startPhase_;
    std::vector<std::unique_ptr<transport::Buffer>> sendRingDataBufs_;
    std::unique_ptr<transport::Buffer> recvRingNotificationBuf_;

    std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendCrossGroupDataBufs_;
    std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendCrossGroupNotificationBufs_;
    std::unordered_map<int, std::unique_ptr<transport::Buffer>> recvCrossGroupNotificationBufs_;
  };


  std::vector<ProxyNode> proxyNodes_;

  void setupProxy(ProxyNode& proxy, grid::Node& original, bool requestNotification);
  void proxyInGroupReduceScatter();
  void proxyInGroupAllGather();
  void proxyCrossGroupAllGather();

  struct AllGatherRequest{
    int rank;
    int offset;
    int length;
  };

  struct Status {
    Phase phase;
    int round;
    bool requestData;
    bool requestNotification;
  };


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
    std::cout << std::endl << std::endl;
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
        exit(0);
      }
    }

  }
};

} // namespace gloo
