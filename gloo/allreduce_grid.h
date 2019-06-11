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

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

template <typename T>
class AllreduceGrid : public Algorithm {
 public:
  AllreduceGrid(
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

    for (int i = 0; i < groups_ - 1; i++) {
      copyInbox_.push_back( static_cast<T*>(malloc(reduceBytes_)) );
    }

    if (this->contextSize_ == 1) {
      return;
    }

    auto& leftPair = this->getPair(leftPairRank_);
    auto& rightPair = this->getPair(rightPairRank_);

    for (int i = 0; i < 2; i++) {
      auto slot = this->context_->nextSlot();
      // Buffer to send to (rank+1).
      sendDataBuf_[i] =
          rightPair->createSendBuffer(slot, ptrs_[0], bytes_);
      // Buffer that (rank-1) writes to.
      recvDataBuf_[i] =
          leftPair->createRecvBuffer(slot, inbox_[i], chunkBytes_);
    }

    // copy buffer
    for (int i = 0; i < groups_ - 1; i ++) {
      auto slot = this->context_->nextSlot();
      auto& copyPair1 = this->getPair(copyPairs_[i]);
      sendCopyDataBuf_.push_back(
          copyPair1->createSendBuffer(slot, ptrs_[0], bytes_) );
      auto& copyPair2 = this->getPair(copyPairs_[groups - 2 - i]);
      recvCopyDataBuf_.push_back(
          copyPair2->createRecvBuffer(slot, copyInbox_[i], reduceBytes_) );
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

  virtual ~AllreduceGrid() {
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

    std::cout << "---- Starting initial copying ----" << std::endl;
    for (int i = 0; i < groups_ - 1; i++) {
      auto chunkOffset = (groupId_ + i + 1) % groups_;
      auto offset = chunkOffset * reduceCount_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      sendCopyDataBuf_[i]->send(offset * sizeof(T), length * sizeof(T));
    }

    for (int i = groups_ - 2; i >= 0; i--) {
      auto offset = reduceOffset_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      recvCopyDataBuf_[i]->waitRecv();
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], copyInbox_[i], length);
      }
    }

    syncRow();

    {
      auto offset = reduceOffset_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      sendCopyDataBuf_[0]->send(offset * sizeof(T), length * sizeof(T));
    }

    {
      auto offset = ((groupId_ + groups_ - 1) % groups_) * reduceCount_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      recvCopyDataBuf_[0]->waitRecv();
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], copyInbox_[0], length);
      }
    }

    syncRow();

    std::cout << "---- Starting in-group allreduce phase 1 ----" << std::endl;

    // Kick off copying initial chunks
    copyChunkAtOffset(2 * this->groupRank_);
    copyChunkAtOffset(2 * this->groupRank_ + 1);


    // Start with reduction of previously copied chunk
    for (int round = 2; round < chunks_; round++) {

      auto chunkOffset = ((2 * this->groupRank_) - (round & ~0x1) +
          (round & 0x1) + chunks_) %
          chunks_;
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

      // Wait for inbox write to complete
      recvDataBuf_[chunkOffset & 1]->waitRecv();

      // Reduce
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], inbox_[chunkOffset & 1], length);
      }

      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendNotificationBuf_->send();

      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
      recvNotificationBuf_->waitRecv();

      syncRow();

      // Copy accumulated chunk
      copyChunkAtOffset(chunkOffset);
    }


    std::cout << "---- Starting in-group allreduce phase 2 ----" << std::endl;

    // Second pass around the ring to broadcast result.
    // End at chunks_-2 since that's where the accumulation
    // stopped in the previous set of rounds.
    for (int round = 0; round < (chunks_ - 2); round++) {

      auto chunkOffset = ((2 * this->groupRank_) - (round & ~0x1) +
          (round & 0x1) + chunks_) %
          chunks_;
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

      // Wait for inbox write to complete
      recvDataBuf_[chunkOffset & 1]->waitRecv();

      // Copy
      if (length > 0) {
        memcpy(&ptrs_[0][offset], inbox_[chunkOffset & 1], length * sizeof(T));
      }

      // Skip copying in the last two rounds
      if (round < (chunks_ - 4)) {
        // Send notification to node on the left that
        // this node is ready for an inbox write.
        sendNotificationBuf_->send();

        // Wait for notification from node on the right
        // to be sure this node can start an inbox write.
        recvNotificationBuf_->waitRecv();

        // Copy accumulated chunks
        copyChunkAtOffset(chunkOffset);
      }
    }

    // Final barrier to make sure every node has finished
    // Otherwise, a second all reduce call might interfere
    // with one that it still in progress on some nodes.
    sendNotificationBuf_->send();
    recvNotificationBuf_->waitRecv();


    std::cout << "---- Starting final copying ----" << std::endl;
    for (int i = 0; i < groups_ - 1; i++) {
      auto offset = reduceOffset_;
      auto length = reduceCount_;
      if (offset + length > count_) {
        length = count_ - offset;
      }
      sendCopyDataBuf_[i]->send(offset * sizeof(T), length * sizeof(T));
    }


    for (int i = 0; i < groups_ - 1; i++) {
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

    syncRow();

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }

 protected:
  void syncRow() {
    for (int i = 0; i < groups_ - 1; i++) {
      sendCopyNotificationBuf_[i]->send();
    }

    for (int i = 0; i < groups_ - 1; i++) {
      recvCopyNotificationBuf_[i]->waitRecv();
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
    sendDataBuf_[chunkOffset & 0x1]->send(
        offset * sizeof(T), length * sizeof(T));

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
  int groupRank_;
  int groupSize_;
  int groupId_;
  int groups_;

  T* inbox_[2];
  std::unique_ptr<transport::Buffer> sendDataBuf_[2];
  std::unique_ptr<transport::Buffer> recvDataBuf_[2];

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;

  std::vector<int> copyPairs_;
  std::vector<T*> copyInbox_;
  std::vector<std::unique_ptr<transport::Buffer>> sendCopyDataBuf_;
  std::vector<std::unique_ptr<transport::Buffer>> recvCopyDataBuf_;

  std::vector<std::unique_ptr<transport::Buffer>> sendCopyNotificationBuf_;
  std::vector<std::unique_ptr<transport::Buffer>> recvCopyNotificationBuf_;
};

} // namespace gloo
