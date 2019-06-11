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
#include <thread>

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

template <typename T>
class AllreduceRingChunked : public Algorithm {
 public:
  AllreduceRingChunked(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum,
      const int groups = 2)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count_ * sizeof(T)),
        fn_(fn),
        groups_(groups){
    printAddr();
    groupSize_ = contextSize_ / groups_;
    groupId_ = contextRank_ / groupSize_;
    groupRank_ = contextRank_ - groupId_ * groupSize_;

    for (int i = 0; i < groups_ - 1; i++) {
      copyPairs_.push_back((contextRank_ + groupSize_ * (i+1)) % contextSize_);
    }

    leftPairRank_ = (groupSize_ + groupRank_ - 1) % groupSize_ + groupId_ * groupSize_;
    rightPairRank_ = (groupRank_ + 1) % groupSize_ + groupId_ * groupSize_;

    // Use chunks of no less than 1024 bytes (256 * sizeof(float))
    // TODO: check chunkSize_
    constexpr unsigned long minSize = 1;
    reduceCount_ = (count_ + groups_ - 1) / groups_;
    reduceBytes_ = reduceCount_ * sizeof(T);
    reduceOffset_ = groupId_ * reduceCount_;

    chunks_ = this->groupSize_ * 2;
    chunkSize_ = std::max(minSize, (reduceCount_ + chunks_ - 1) / chunks_);
    chunkBytes_ = chunkSize_ * sizeof(T);

    // Allocate inboxes
    for (int i = 0; i < 2; i++) {
      inbox_[i] = static_cast<T*>(malloc(bytes_));
    }

    for (int i = 0; i < groups_; i++) {
      copyInbox_.push_back( static_cast<T*>(malloc(reduceBytes_)) );
    }

    if (this->contextSize_ == 1) {
      return;
    }

    auto& leftPair = this->getPair(leftPairRank_);
    auto& rightPair = this->getPair(rightPairRank_);

    for (int i = 0; i < 2; i++) {
      auto slot = this->context_->nextSlot(1);
      // Buffer to send to (rank+1).
      sendDataBuf_[i] =
        rightPair->createSendBuffer(slot, ptrs_[0], bytes_);
      // Buffer that (rank-1) writes to.
      recvDataBuf_[i] =
        leftPair->createRecvBuffer(slot, inbox_[i], chunkBytes_);
    }

    // copy buffer
//    sendCopyDataBuf_ = new std::unique_ptr<transport::Buffer>[groups_ - 1];
//    recvCopyDataBuf_ = new std::unique_ptr<transport::Buffer>[groups_ - 1];

    for (int i = 0; i < groups_ - 1; i ++) {
      auto slot = this->context_->nextSlot();
      auto& copyPair1 = this->getPair(copyPairs_[i]);
      sendCopyDataBuf_.push_back(
          copyPair1->createSendBuffer(slot, ptrs_[0], bytes_) );
      auto& copyPair2 = this->getPair(copyPairs_[groups - 2 - i]);
      recvCopyDataBuf_.push_back(
          copyPair2->createRecvBuffer(slot, copyInbox_[i], reduceBytes_) );
    }

    {
      auto slot = this->context_->nextSlot();
      auto& copyPair1 = this->getPair(copyPairs_[0]);
      sendCopyDataBuf_.push_back(
          copyPair1->createSendBuffer(slot, ptrs_[0], bytes_) );
      auto& copyPair2 = this->getPair(copyPairs_[groups - 2]);
      recvCopyDataBuf_.push_back(
          copyPair2->createRecvBuffer(slot, copyInbox_[groups - 1], reduceBytes_) );
    }


    // Dummy buffers for localized barrier.
    // Before sending to the right, we only need to know that the node
    // on the right is done using the inbox that's about to be written
    // into. No need for a global barrier.
    auto notificationSlot = this->context_->nextSlot();
    sendNotificationBuf_ =
        leftPair->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
        rightPair->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));

    // copy notification
    for (int i = 0; i < groups_ - 1; i ++) {
      auto slot = this->context_->nextSlot();
      auto& copyPair1 = this->getPair(copyPairs_[i]);
      sendCopyNotificationBuf_.push_back(
          copyPair1->createSendBuffer(slot, &dummy_, sizeof(dummy_)) );
      auto& copyPair2 = this->getPair(copyPairs_[groups - 2 - i]);
      recvCopyNotificationBuf_.push_back(
          copyPair2->createRecvBuffer(slot, &dummy_, sizeof(dummy_)) );
    }

  }

  virtual ~AllreduceRingChunked() {
    for (int i = 0; i < 2; i++) {
      if (inbox_[i] != nullptr) {
        free(inbox_[i]);
      }
    }
    for (int i = 0; i < groups_ - 1; i++) {
      free(copyInbox_[i]);
    }
  }

  void run() {
    // Reduce specified pointers into ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      fn_->call(ptrs_[0], ptrs_[i], count_);
    }

    if (this->contextSize_ == 1) {
      // Broadcast ptrs_[0]
      for (int i = 1; i < ptrs_.size(); i++) {
        memcpy(ptrs_[i], ptrs_[0], bytes_);
      }
      return;
    }

    bool success;

    std::cout << "---- Starting initial copying ----" << std::endl;
    for (int i = 0; i < groups_ - 1; i++) {
      auto chunkOffset = (groupId_ + i + 1) % groups_;
      auto offset = chunkOffset * reduceCount_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      success = sendCopyDataBuf_[i]->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) repairGroupId_ = (groupId_ + 1 + i) % groups_;
    }

    for (int i = groups_ - 2; i >= 0; i--) {
      if (repairGroupId_ == (groupId_ + groups_ - 1 - i) % groups_) continue;
      success = recvCopyDataBuf_[i]->tryWaitRecv();
      if (!success) repairGroupId_ = (groupId_ + groups_ - 1 - i) % groups_;
    }


    if ((groupId_ + 1) % groups_ != repairGroupId_) {
      auto offset = reduceOffset_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      success = sendCopyDataBuf_[groups_-1]->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) repairGroupId_ = (groupId_ + 1) % groups_;
    }

    if ((groupId_ + groups_ - 1) % groups_ != repairGroupId_) {
      success = recvCopyDataBuf_[groups_-1]->tryWaitRecv();
      if (!success) repairGroupId_ = (groupId_ + groups_ - 1) % groups_;
    }


    std::cout << "sync row" << std::endl;
    round_ = 1;

    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + 1 + i) % groups_) continue;
      success = sendCopyNotificationBuf_[i]->trySend();
      if (!success) repairGroupId_ = (groupId_ + 1 + i) % groups_;
    }

    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + groups_ - 1 - i) % groups_) continue;
      success = recvCopyNotificationBuf_[i]->tryWaitRecv();
      if (!success) repairGroupId_ = (groupId_ + groups_ - 1 - i) % groups_;
    }

    std::cout << "reduce copy data" << std::endl;
    for (int i = groups_ - 2; i >= 0; i--) {
      if (repairGroupId_ == (groupId_ + groups_ - 1 - i) % groups_) continue;
      auto offset = reduceOffset_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], copyInbox_[i], length);
      }
    }

    if ((groupId_ + groups_ - 1) % groups_ != repairGroupId_) {
      auto offset = ((groupId_ + groups_ - 1) % groups_) * reduceCount_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], copyInbox_[groups_-1], length);
      }
    }

    printData();

    if (repairGroupId_ >= 0) {
      repairRow(repairGroupId_);
      repairGroupId_ = -1;
    }

    std::cout << "---- Starting in-group allreduce phase 1 ----" << std::endl;

    // Kick off copying initial chunks
    copyChunkAtOffset(2 * this->groupRank_);
    copyChunkAtOffset(2 * this->groupRank_ + 1);


    // Start with reduction of previously copied chunk
    for (round_ = 2; round_ < chunks_; round_++) {
      std::cout << "-- Starting phase 1 round " << round_ << " --" << std::endl;

      auto chunkOffset = ((2 * this->groupRank_) - (round_ & ~0x1) +
                          (round_ & 0x1) + chunks_) %
          chunks_;
      int offset, length;
      std::tie(offset, length) = getChunkLoc(chunkOffset);

      if (!leftPairFailed_) {
        std::cout << "Receiving chunk " << chunkOffset << " from " << leftPairRank_ << std::endl;
        // Wait for inbox write to complete
        success = recvDataBuf_[chunkOffset & 1]->tryWaitRecv();
        if (!success) {
          lostChunkOffset1_ = chunkOffset;
          lostChunkOffset2_ = nextChunkOffset(chunkOffset);
          leftPairNeedRepair_ = true;
          recoverLeftPair();
        }
      }

      if (leftPairFailed_) {
        auto originalLeftPairGroupRank = (groupSize_ + groupRank_ - 1) % groupSize_;
        auto chunkOffset2 = ((2 * originalLeftPairGroupRank) - (round_ & ~0x1) +
            (round_ & 0x1) + chunks_) %
            chunks_;
        int offset2, length2;
        std::tie(offset2, length2) = getChunkLoc(chunkOffset2);


        std::cout << "Receiving chunk " << chunkOffset2 << " from " << leftPairRank_ << " (backup)" << std::endl;
        recvDataBuf_[chunkOffset2 & 1]->waitRecv();
        if (length2 > 0) {
          fn_->call(&backupPtrs_[0][offset2-reduceOffset_], inbox_[chunkOffset2 & 1], length2);
        }
      }

      if (!leftPairFailed_) {
        // Reduce
        if (length > 0) {
          fn_->call(&ptrs_[0][offset], inbox_[chunkOffset & 1], length);
        }
      } else {
        if (length > 0) {
          fn_->call(&ptrs_[0][offset], &backupPtrs_[0][offset-reduceOffset_], length);
        }
      }


      std::cout << "Sending notification to " << leftPairRank_ << std::endl;
      // Send notification to node on the left that
      // this node is ready for an inbox write.
      success = sendNotificationBuf_->trySend();
      if (!success) {
        lostChunkOffset1_ = nextChunkOffset(chunkOffset);
        leftPairNeedRepair_ = true;
        recoverLeftPair();
        sendNotificationBuf_->send();
      }

      syncRow();


      if (rightPairNeedRepair_) {
        recoverRightPair();
        rightPairNeedRepair_ = false;
      }

      std::cout << "Waiting notification from " << rightPairRank_ << std::endl;
      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
      success = recvNotificationBuf_->tryWaitRecv();
      if (!success) {
        inflightChunkOffset1_ = prevChunkOffset(chunkOffset);
        inflightChunkOffset2_ = chunkOffset;
        rightPairNeedRepair_ = true;
      }


      // Copy accumulated chunk
      if (!rightPairNeedRepair_) {
        std::cout << "Sending chunk " << chunkOffset << " to " << rightPairRank_ << std::endl;
        copyChunkAtOffset(chunkOffset);
      }

      printData();
    }

    std::cout << "---- Starting in-group allreduce phase 2 ----" << std::endl;

    // Second pass around the ring to broadcast result.
    // End at chunks_-2 since that's where the accumulation
    // stopped in the previous set of rounds.
    for (; round_ < (chunks_ * 2 - 2); round_++) {
      std::cout << "-- Starting phase 2 round " << (round_ - chunks_) << " --" << std::endl;
      testRepair(round_);

      auto chunkOffset = ((2 * this->groupRank_) - ((round_ - chunks_) & ~0x1) +
                          ((round_ - chunks_) & 0x1) + chunks_) %
          chunks_;
      int offset, length;
      std::tie(offset, length) = getChunkLoc(chunkOffset);

      if (!leftPairFailed_) {
        std::cout << "Receiving chunk " << chunkOffset << " from " << leftPairRank_ << std::endl;
        // Wait for inbox write to complete
        success = recvDataBuf_[chunkOffset & 1]->tryWaitRecv();
        if (!success) {
          lostChunkOffset1_ = chunkOffset;
          lostChunkOffset2_ = nextChunkOffset(chunkOffset);
          leftPairNeedRepair_ = true;
          recoverLeftPair();
        }
      }

      if (leftPairFailed_) {
        auto originalLeftPairGroupRank = (groupSize_ + groupRank_ - 1) % groupSize_;
        auto chunkOffset2 = ((2 * originalLeftPairGroupRank) - ((round_ - chunks_) & ~0x1) +
            ((round_ - chunks_) & 0x1) + chunks_) %
            chunks_;
        int offset2, length2;
        std::tie(offset2, length2) = getChunkLoc(chunkOffset2);

        std::cout << "Receiving chunk " << chunkOffset2 << " from " << leftPairRank_ << " (backup)" << std::endl;
        recvDataBuf_[chunkOffset2 & 1]->waitRecv();
        if (length2 > 0) {
          memcpy(&backupPtrs_[0][offset2-reduceOffset_], inbox_[chunkOffset2 & 1], length2 * sizeof(T));
        }
      }

      if (!leftPairFailed_) {
        // Copy
        if (length > 0) {
          memcpy(&ptrs_[0][offset], inbox_[chunkOffset & 1], length * sizeof(T));
        }
      } else {
        if (length > 0) {
          memcpy(&ptrs_[0][offset], &backupPtrs_[0][offset-reduceOffset_], length * sizeof(T));
        }
      }


      // Send notification to node on the left that
      // this node is ready for an inbox write.
      std::cout << "Sending notification to " << leftPairRank_ << std::endl;
      success = sendNotificationBuf_->trySend();
      if (!success) {
        lostChunkOffset1_ = nextChunkOffset(chunkOffset);
        leftPairNeedRepair_ = true;
        recoverLeftPair();
        sendNotificationBuf_->send();
      }

      syncRow();

      if (rightPairNeedRepair_) {
        recoverRightPair();
        rightPairNeedRepair_ = false;
      }

      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
      std::cout << "Waiting notification from " << rightPairRank_ << std::endl;
      success = recvNotificationBuf_->tryWaitRecv();
      if (!success) {
        inflightChunkOffset1_ = prevChunkOffset(chunkOffset);
        inflightChunkOffset2_ = chunkOffset;
        rightPairNeedRepair_ = true;
      }

      // Skip copying in the last two rounds
      if (round_ < (chunks_ * 2 - 4)) {
        // Copy accumulated chunks
        if (!rightPairNeedRepair_) {
          std::cout << "Sending chunk " << chunkOffset << " to " << rightPairRank_ << std::endl;
          copyChunkAtOffset(chunkOffset);
        }
      }

      printData();
    }

    if (rightPairNeedRepair_) {
      recoverRightPair();
      rightPairNeedRepair_ = false;
    }

    std::cout << "---- Starting final copying ----" << std::endl;
    for (int i = 0; i < groups_ - 1; i++) {
      if (copyPairs_[i] == failedRank_) continue;
      auto offset = reduceOffset_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      success = sendCopyDataBuf_[i]->trySend(offset * sizeof(T), length * sizeof(T));
      if (!success) repairGroupId_ = (groupId_ + 1 + i) % groups_;
    }


    if (leftPairFailed_) {
      for (int i = 0; i < groups_ - 1; i++) {
        auto offset = reduceOffset_;
        auto length = reduceCount_;
        if (offset + length > count_) {
          length = count_ - offset;
        }
        backupSendCopyDataBuf_[i]->send(offset * sizeof(T), length * sizeof(T));
      }
    }

    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + 1 + i) % groups_) continue;
      auto chunkOffset = (groupId_ + i + 1) % groups_;
      auto offset = chunkOffset * reduceCount_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      success = recvCopyDataBuf_[groups_ - 2 - i]->tryWaitRecv();
      if (!success) repairGroupId_ = (groupId_ + 1 + i) % groups_;
      if (success && length > 0) {
        memcpy(&ptrs_[0][offset], copyInbox_[groups_ - 2 - i], length * sizeof(T));
      }
    }

    printData();


    // Final barrier to make sure every node has finished
    // Otherwise, a second all reduce call might interfere
    // with one that it still in progress on some nodes.

    std::cout << "Sending notification to " << leftPairRank_ << std::endl;
    success = sendNotificationBuf_->trySend();
    if (!success) {
      recoverLeftPair();
      sendNotificationBuf_->send();
    }

    std::cout << "sync row" << std::endl;
    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + 1 + i) % groups_) continue;
      success = sendCopyNotificationBuf_[i]->trySend();
      if (!success) repairGroupId_ = (groupId_ + 1 + i) % groups_;
    }

    if (leftPairFailed_) {
      for (int i = 0; i < groups_ - 1; i++)
        backupSendCopyNotificationBuf_[i]->send();
    }

    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + groups_ - 1 - i) % groups_) continue;
      success = recvCopyNotificationBuf_[i]->tryWaitRecv();
      if (!success) repairGroupId_ = (groupId_ + groups_ - 1 - i) % groups_;
    }

    if (leftPairFailed_) {
      for (int i = 0; i < groups_ - 1; i++)
        backupRecvCopyNotificationBuf_[i]->waitRecv();
    }

    if (repairGroupId_ >= 0) {
      repairRow(repairGroupId_);
      for (int i = 0; i < groups_ - 1; i++) {
        if (repairGroupId_ != (groupId_ + groups_ - 1 - i) % groups_) continue;
        auto chunkOffset = (groupId_ + i + 1) % groups_;
        auto offset = chunkOffset * reduceCount_;
        auto length = reduceCount_;
        if (offset + length > count_) {
          length = count_ - offset;
        }
        recvCopyDataBuf_[groups_ - 2 - i]->waitRecv();
        if (length > 0) {
          memcpy(&ptrs_[0][offset], copyInbox_[groups_ - 2 - i], length * sizeof(T));
        }
      }
      repairGroupId_ = -1;
    }

    std::cout << "Waiting notification from " << rightPairRank_ << std::endl;
    success = recvNotificationBuf_->tryWaitRecv();
    if (!success) {
      recoverRightPair();
      recvNotificationBuf_->waitRecv();
    }


    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }

 protected:
  void syncRow() {
    std::cout << "sync row" << std::endl;
    bool success;
    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + 1 + i) % groups_) continue;
      success = sendCopyNotificationBuf_[i]->trySend();
      if (!success) repairGroupId_ = (groupId_ + 1 + i) % groups_;
    }
    if (leftPairFailed_) {
      for (int i = 0; i < groups_ - 1; i++)
        backupSendCopyNotificationBuf_[i]->send();
    }

    if (repairGroupId_ >= 0) {
      std::cout << "Repair group " << repairGroupId_ << std::endl;
      repairRow(repairGroupId_);
    }

    for (int i = 0; i < groups_ - 1; i++) {
      if (repairGroupId_ == (groupId_ + groups_ - 1 - i) % groups_) {
        continue;
      }
      success = recvCopyNotificationBuf_[i]->tryWaitRecv();
      if (!success) {
        repairGroupId_ = (groupId_ + groups_ - 1 - i) % groups_;
        std::cout << "Repair group " << repairGroupId_ << std::endl;
        repairRow(repairGroupId_);
      }
    }
    if (leftPairFailed_) {
      for (int i = 0; i < groups_ - 1; i++)
        backupRecvCopyNotificationBuf_[i]->waitRecv();
    }
    repairGroupId_ = -1;
  }

  void repairRow(int repairGroupId) {
    failedRank_ = repairGroupId * groupSize_ + groupRank_;
    auto backupRank = repairGroupId * groupSize_ + (groupRank_ + 1) % groupSize_;
    auto& backupPair = context_->getPair(backupRank);

    auto slotOffset = this->context_->nextSlot(groups_ - 1);
    auto slot = slotOffset + (groupId_ + groups_ - repairGroupId) % groups_ - 1;
    std::unique_ptr<transport::Buffer> sendBackupBuf = backupPair->createSendBuffer(slot, ptrs_[0], bytes_);

    auto offset = repairGroupId * reduceCount_;
    auto length = reduceCount_;
    if (offset + length > count_) {
      length = count_ - offset;
    }
    sendBackupBuf->send(offset * sizeof(T), length * sizeof(T));
    sendBackupBuf->waitSend();

    std::cout << "[R] send backup to " << backupRank << " through slot " << slot << std::endl;

    // adjust buffers
    auto failedSendIdx = (repairGroupId + groups_ - groupId_) % groups_ - 1;
    auto failedRecvIdx = groups_ - 2 - failedSendIdx;
    slotOffset = this->context_->nextSlot(groups_ - 1);
    slot = slotOffset + failedRecvIdx;
    recvCopyDataBuf_[failedRecvIdx] = backupPair->createRecvBuffer(slot, copyInbox_[failedRecvIdx], reduceBytes_);

    slotOffset = this->context_->nextSlot(groups_ - 1);
    slot = slotOffset + failedSendIdx;
    sendCopyNotificationBuf_[failedSendIdx] = backupPair->createSendBuffer(slot, &round_, sizeof(dummy_));
    slot = slotOffset + failedRecvIdx;
    recvCopyNotificationBuf_[failedRecvIdx] = backupPair->createRecvBuffer(slot, &backupPairRound_, sizeof(dummy_));

    sendCopyNotificationBuf_[failedSendIdx]->send();
    recvCopyNotificationBuf_[failedRecvIdx]->waitRecv();


    if (backupPairRound_ < round_) {
      std::cout << "[R] resync row to round " << backupPairRound_ << std::endl;
      for (int j = 0; j < round_ - backupPairRound_; j++) {
        sendCopyNotificationBuf_[failedSendIdx]->send();
        recvCopyNotificationBuf_[failedRecvIdx]->waitRecv();
      }
    }

    sendCopyNotificationBuf_[failedSendIdx]->send();
    recvCopyNotificationBuf_[failedRecvIdx]->waitRecv();

  }

  void recoverLeftPair() {
    for (int i = 0; i < groups_ - 1; i++) {
      backupPtrs_.push_back( static_cast<T*>(malloc(reduceBytes_)) );
    }

    auto slotOffset = this->context_->nextSlot(groups_ - 1);
    std::vector<std::unique_ptr<transport::Buffer>> recvBackupBuf;
    for (int i = 1; i < groups_; i++) {
      auto backupRank = (leftPairRank_ + groupSize_ * i) % contextSize_;
      auto& backupPair = context_->getPair(backupRank);
      backupCopyPairRank_.push_back(backupRank);

      auto slot = slotOffset + i - 1;
      recvBackupBuf.push_back(backupPair->createRecvBuffer(slot, backupPtrs_[i-1], reduceBytes_));
      std::cout << "[R] recv backup from " << backupRank << " through slot " << slot << std::endl;
    }

    // backup buffers
    for (int i = 0; i < groups_ - 1; i ++) {
      auto slot = this->context_->nextSlot();
      auto& copyPair1 = this->getPair(backupCopyPairRank_[i]);
      backupSendCopyDataBuf_.push_back(
          copyPair1->createSendBuffer(slot, ptrs_[0], bytes_) );
    }

    backupRowRound_ = static_cast<int*>(malloc(sizeof(int) * (groups_ - 1)));
    for (int i = 0; i < groups_ - 1; i ++) {
      auto slot = this->context_->nextSlot();
      auto& copyPair1 = this->getPair(backupCopyPairRank_[i]);
      backupSendCopyNotificationBuf_.push_back(
          copyPair1->createSendBuffer(slot, &round_, sizeof(dummy_)) );
      auto& copyPair2 = this->getPair(backupCopyPairRank_[groups_ - 2 - i]);
      backupRecvCopyNotificationBuf_.push_back(
          copyPair2->createRecvBuffer(slot, &backupRowRound_[i], sizeof(dummy_)) );
    }

    for (int i = 0; i < groups_ - 1; i++)
      backupSendCopyNotificationBuf_[i]->send();
    for (int i = 0; i < groups_ - 1; i++)
      backupRecvCopyNotificationBuf_[i]->waitRecv();

    if (round_ == chunks_ * 2 - 2) {
      for (int i = 0; i < groups_ - 1; i++) {
        auto offset = reduceOffset_;
        auto length = reduceCount_;
        if (offset + length > count_) {
          length = count_ - offset;
        }
        backupSendCopyDataBuf_[i]->send(offset * sizeof(T), length * sizeof(T));
      }
    }

    for (int i = 0; i < groups_ - 1; i++) {
      if (backupRowRound_[groups_ - 2 - i] < round_) {
        std::cout << "[R] resync row to round " << backupRowRound_[groups_ - 2 - i] << std::endl;
        for (int j = 0; j < round_ - backupRowRound_[groups_ - 2 - i]; j++)
          backupSendCopyNotificationBuf_[i]->send();
      }
    }
    for (int i = 0; i < groups_ - 1; i++) {
      if (backupRowRound_[i] < round_) {
        for (int j = 0; j < round_ - backupRowRound_[i]; j++)
          backupRecvCopyNotificationBuf_[i]->waitRecv();
      }
    }

    for (int i = 0; i < groups_ - 1; i++) {
      recvBackupBuf[i]->waitRecv();
    }

    for (int i = 1; i < backupPtrs_.size(); i++) {
      fn_->call(backupPtrs_[0], backupPtrs_[i], reduceCount_);
    }

    // update left pair rank
    leftPairFailed_ = true;
    leftPairRank_ = (groupRank_ + groupSize_ - 2) % groupSize_ + groupId_ * groupSize_;
    auto& leftPair = context_->getPair(leftPairRank_);

    for (int i = 0; i < 2; i++) {
      auto slot = recvDataBuf_[i]->getSlot();
      recvDataBuf_[i] =
          leftPair->createRecvBuffer(slot, inbox_[i], chunkBytes_);
    }

    auto notificationSlot = sendNotificationBuf_->getSlot();
    sendNotificationBuf_ =
        leftPair->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));

    // repair backup ptrs
    auto originalLeftPairGroupRank = (groupSize_ + groupRank_ - 1) % groupSize_;
    if (round_ == 2 || round_ == 3) {
      lostChunkOffset1_ = -1;
    }
    if (round_ == 2) {
      lostChunkOffset2_ = -1;
    }

    std::cout << "[R] repairing chunks at " << lostChunkOffset1_ << " " << lostChunkOffset2_ << std::endl;
    auto syncRoundBuf = leftPair->createRecvBuffer(notificationSlot + 1, &leftPairRound_, sizeof(leftPairRound_));

    dummy_ = lostChunkOffset1_;
    sendNotificationBuf_->send();

    int offset, length;
    if (lostChunkOffset1_ >= 0) {
      std::tie(offset, length) = getChunkLoc(lostChunkOffset1_);
      recvDataBuf_[lostChunkOffset1_ & 1]->waitRecv();
      if (length > 0) {
        if (round_ < chunks_ + 2) {
          fn_->call(&backupPtrs_[0][offset - reduceOffset_], inbox_[lostChunkOffset1_ & 1], length);
        } else {
          memcpy(&backupPtrs_[0][offset - reduceOffset_], inbox_[lostChunkOffset1_ & 1], length * sizeof(T));
        }
      }
    }
    syncRoundBuf->waitRecv();

    dummy_ = lostChunkOffset2_;
    sendNotificationBuf_->send();
    if (lostChunkOffset2_ >= 0) {
      std::tie(offset, length) = getChunkLoc(lostChunkOffset2_);
      recvDataBuf_[lostChunkOffset2_ & 1]->waitRecv();
      if (length > 0) {
        if (round_ < chunks_ + 1) {
          fn_->call(&backupPtrs_[0][offset - reduceOffset_], inbox_[lostChunkOffset2_ & 1], length);
        } else {
          memcpy(&backupPtrs_[0][offset - reduceOffset_], inbox_[lostChunkOffset2_ & 1], length);
        }
      }
    }
    syncRoundBuf->waitRecv();

    dummy_ = round_;
    sendNotificationBuf_->send();

    if (leftPairRound_ < round_) {
      std::cout << "[R] resync left from round " << leftPairRound_ << std::endl;
      sendNotificationBuf_->send();
    }

  }

  void recoverRightPair() {
    // update right pair rank
    rightPairFailed_ = true;
    rightPairRank_ = (groupRank_ + 2) % groupSize_ + groupId_ * groupSize_;
    auto& rightPair = context_->getPair(rightPairRank_);

    for (int i = 0; i < 2; i++) {
      auto slot = sendDataBuf_[i]->getSlot();
      sendDataBuf_[i] =
          rightPair->createSendBuffer(slot, ptrs_[0], bytes_);
    }

    auto notificationSlot = recvNotificationBuf_->getSlot();
    recvNotificationBuf_ =
        rightPair->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));

    auto syncRoundBuffer = rightPair->createSendBuffer(notificationSlot + 1, &round_, sizeof(round_));

    // repair backup ptrs

    recvNotificationBuf_->waitRecv();
    auto chunkOffset1 = dummy_;
    int offset, length;
    if (chunkOffset1 >= 0) {
      std::tie(offset, length) = getChunkLoc(chunkOffset1);
      copyChunkAtOffset(chunkOffset1);
    }
    syncRoundBuffer->send();

    recvNotificationBuf_->waitRecv();
    auto chunkOffset2 = dummy_;
    if (chunkOffset2 >= 0) {
      std::tie(offset, length) = getChunkLoc(chunkOffset2);
      copyChunkAtOffset(chunkOffset2);
    }
    std::cout << "[R] repairing chunks at " << chunkOffset1 << " " << chunkOffset2 << std::endl;


    // sync round
    syncRoundBuffer->send();

    recvNotificationBuf_->waitRecv();
    rightPairRound_ = dummy_;


    // resend inflight
    int lostChunk1 = nextChunkOffset(chunkOffset2);
    if (lostChunk1 >= 0) {
      std::cout << "[R] resending chunk at " << lostChunk1 << std::endl;
      copyChunkAtOffset(lostChunk1);
    } else if (inflightChunkOffset1_ >= 0) {
      std::cout << "[R] resending chunk at " << inflightChunkOffset1_ << std::endl;
      copyChunkAtOffset(inflightChunkOffset1_);
      inflightChunkOffset1_ = -1;
    }
    int lostChunk2 = nextChunkOffset(lostChunk1);
    if (lostChunk2 >= 0) {
      std::cout << "[R] resending chunk at " << lostChunk2 << std::endl;
      copyChunkAtOffset(lostChunk2);
    } else if (inflightChunkOffset2_ >= 0) {
      std::cout << "[R] resending chunk at " << inflightChunkOffset2_ << std::endl;
      copyChunkAtOffset(inflightChunkOffset2_);
      inflightChunkOffset2_ = -1;
    }


    if (rightPairRound_ < round_) {
      std::cout << "[R] resync right from round " << rightPairRound_ << std::endl;
      recvNotificationBuf_->waitRecv();

      if (inflightChunkOffset1_ >= 0 && inflightChunkOffset1_ != chunkOffset1 && inflightChunkOffset1_ != chunkOffset2
      && inflightChunkOffset1_ != lostChunk1 && inflightChunkOffset1_ != lostChunk2) {
        std::cout << "[R] resending chunk at " << inflightChunkOffset1_ << std::endl;
        copyChunkAtOffset(inflightChunkOffset1_);
      }
      if (inflightChunkOffset2_ >= 0 && inflightChunkOffset2_ != lostChunk1 && inflightChunkOffset2_ != lostChunk2) {
        std::cout << "[R] resending chunk at " << inflightChunkOffset2_ << std::endl;
        copyChunkAtOffset(inflightChunkOffset2_);
      }
    }

  }

  std::tuple<int, int> getChunkLoc(int chunkOffset) {
    auto offset = chunkOffset * chunkSize_;
    auto length = chunkSize_;
    if (offset + length <= reduceCount_) {
    } else if (offset < reduceCount_) {
      length = reduceCount_ - offset;
    } else {
      length = 0;
    }

    offset += reduceOffset_;
    if (offset + length <= count_) {
      // Chunk completely in range, copy full chunk.
    } else if (offset < count_) {
      // Chunk partially in range, copy partial chunk.
      length = count_ - offset;
    } else {
      // Chunk out of range, copy nothing.
      length = 0;
    }
    return std::make_tuple(offset, length);
  }

  int nextChunkOffset(int chunkOffset) {
    if (chunkOffset < 0)
      return -1;
    if (chunkOffset % 2 == 0)
      return chunkOffset + 1;
    else
      return (chunkOffset + chunks_ - 3) % chunks_;
  }

  int prevChunkOffset(int chunkOffset) {
    if (chunkOffset < 0)
      return -1;
    if (chunkOffset % 2 != 0)
      return chunkOffset - 1;
    else
      return (chunkOffset + 3) % chunks_;
  }


  void printData() {
    if (leftPairFailed_) {
      for (int i = 0; i < reduceCount_; i++) {
        std::cout << "backup[" << i << "] = " << backupPtrs_[0][i] << std::endl;
      }
    }
    for (int i = 0; i < count_; i++) {
      if (i < reduceOffset_ || i >= reduceOffset_ + reduceCount_) continue;
      std::cout << "data[" << i << "] = " << ptrs_[0][i] << std::endl;
    }
  }

  void printAddr() {
    for (int i = 0; i < contextSize_; i++) {
      if (i == contextRank_) continue;
      auto& pair = context_->getPair(i);
      std::cout << "Pair " << i << ": " << pair->address().str() << std::endl;
    }
  }

  void testRepair(int round) {
    if (round != -1) return;
    if (contextRank_ == 0) {
      exit(0);
    }
  }


  void copyChunkAtOffset(int chunkOffset) {
    // Populate inbox of next participant in the ring.
    auto offset = (chunkOffset % chunks_) * chunkSize_;
    auto length = chunkSize_;
    if (offset + length <= reduceCount_) {
    } else if (offset < reduceCount_) {
      length = reduceCount_ - offset;
    } else {
      length = 0;
    }
    offset += reduceOffset_;
    if (offset + length <= count_) {
      // Chunk completely in range, copy full chunk.
    } else if (offset < count_) {
      // Chunk partially in range, copy partial chunk.
      length = count_ - offset;
    } else {
      // Chunk out of range, copy _something_.
      // When nothing is put on the wire for empty chunks. @pietern
      // has seen this algorithm hang. This is probably related to the
      // chunk iteration order described in the run function.
      offset = 0;
      length = 1;
    }

    // Initiate write to inbox of node on the right.
    auto success = sendDataBuf_[chunkOffset & 0x1]->trySend(
        offset * sizeof(T), length * sizeof(T));
    if (!success) {
      inflightChunkOffset1_ = prevChunkOffset(chunkOffset);
      inflightChunkOffset2_ = chunkOffset;
      rightPairNeedRepair_ = true;
    }

  }


  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const ReductionFunction<T>* fn_;

  int reduceCount_;
  int reduceBytes_;
  size_t reduceOffset_;

  size_t chunks_;
  size_t chunkSize_;
  size_t chunkBytes_;

  int leftPairRank_;
  int rightPairRank_;
  int copyPairRank_;
  int groupRank_;
  int groupSize_;
  int groupId_;
  int groups_;

  bool copyPairFailed_ = false;
  int copyRightPairRank_;
  bool rightPairFailed_ = false;
  bool leftPairFailed_ = false;
  int failedRank_ = -1;

  T* inbox_[2];
  std::unique_ptr<transport::Buffer> sendDataBuf_[2];
  std::unique_ptr<transport::Buffer> recvDataBuf_[2];

  std::unique_ptr<transport::Buffer> sendDataBuf2_[2];
  std::unique_ptr<transport::Buffer> recvNotificationBuf2_;

  int dummy_ = -1;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;

  std::unique_ptr<transport::Buffer> copySendNotificationBuf2_;
  std::unique_ptr<transport::Buffer> copyRecvNotificationBuf2_;

  std::vector<int> copyPairs_;
  std::vector<T*> copyInbox_;
  std::vector<std::unique_ptr<transport::Buffer>> sendCopyDataBuf_;
  std::vector<std::unique_ptr<transport::Buffer>> recvCopyDataBuf_;

  std::vector<std::unique_ptr<transport::Buffer>> sendCopyNotificationBuf_;
  std::vector<std::unique_ptr<transport::Buffer>> recvCopyNotificationBuf_;

  std::vector<T*> backupPtrs_;
  std::vector<int> backupCopyPairRank_;
  std::vector<std::unique_ptr<transport::Buffer>> backupSendCopyDataBuf_;
  std::vector<std::unique_ptr<transport::Buffer>> backupRecvCopyDataBuf_;

  std::vector<std::unique_ptr<transport::Buffer>> backupSendCopyNotificationBuf_;
  std::vector<std::unique_ptr<transport::Buffer>> backupRecvCopyNotificationBuf_;

  std::thread repairLeftThread;
  bool leftPairNeedRepair_ = false;
  int lostChunkOffset1_ = -1;
  int lostChunkOffset2_ = -1;
  std::thread repairRightThread;
  bool rightPairNeedRepair_ = false;
  int inflightChunkOffset1_ = -1;
  int inflightChunkOffset2_ = -1;
  std::thread repairRowThread;
  int repairGroupId_ = -1;

  int round_ = -1;
  int backupPairRound_ = -1;
  int* backupRowRound_;
  int leftPairRound_ = -1;
  int rightPairRound_ = -1;
};

} // namespace gloo
