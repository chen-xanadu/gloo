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
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count_ * sizeof(T)),
        fn_(fn) {
    // Use chunks of no less than 1024 bytes (256 * sizeof(float))
    constexpr unsigned long minSize = 32768;
    chunks_ = this->contextSize_ * 2;
    chunkSize_ = std::max(minSize, (count_ + chunks_ - 1) / chunks_);
    chunks_ = (count_ + chunkSize_ - 1) / chunkSize_;
    chunkBytes_ = chunkSize_ * sizeof(T);

    // Allocate inboxes
    for (int i = 0; i < 2; i++) {
      inbox_[i] = static_cast<T*>(malloc(bytes_));
    }

    if (this->contextSize_ == 1) {
      return;
    }

    auto& leftPair = this->getLeftPair();
    auto& rightPair = this->getRightPair();
    for (int i = 0; i < 2; i++) {
      auto slot = this->context_->nextSlot();

      // Buffer to send to (rank+1).
      sendDataBuf_[i] =
          rightPair->createSendBuffer(slot, ptrs_[0], bytes_);
      // Buffer that (rank-1) writes to.
      recvDataBuf_[i] =
          leftPair->createRecvBuffer(slot, inbox_[i], chunkBytes_);
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
  }

  virtual ~AllreduceRingChunked() {
    for (int i = 0; i < 2; i++) {
      if (inbox_[i] != nullptr) {
        free(inbox_[i]);
      }
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

    int numRounds = (int)(chunks_ * (this->contextSize_ - 1));

    // Kick off copying initial chunks
    copyChunkAtOffset(0);
    if (chunks_ > 1)
      copyChunkAtOffset(1);

    for (int round = 0; round < numRounds; round++) {

      auto chunkOffset = round % chunks_;
      auto offset = chunkOffset * chunkSize_;
      auto length = chunkSize_;
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
      recvDataBuf_[chunkOffset % 2]->waitRecv();

      // Reduce
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], inbox_[chunkOffset % 2], length);
      }

      if ((chunks_ == 1 && round < numRounds - 1) || round < numRounds - 2) {
        // Send notification to node on the left that
        // this node is ready for an inbox write.
        sendNotificationBuf_->send();

        // Wait for notification from node on the right
        // to be sure this node can start an inbox write.
        recvNotificationBuf_->waitRecv();

        // Copy accumulated chunk
        copyChunkAtOffset((chunkOffset + 2) % chunks_);
      }

    }

    // Final barrier to make sure every node has finished
    // Otherwise, a second all reduce call might interfere
    // with one that it still in progress on some nodes.
    sendNotificationBuf_->send();
    recvNotificationBuf_->waitRecv();

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }

 protected:
  void copyChunkAtOffset(int chunkOffset) {
    // Populate inbox of next participant in the ring.
    auto offset = (chunkOffset % chunks_) * chunkSize_;
    auto length = chunkSize_;
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
    sendDataBuf_[chunkOffset % 2]->send(
        offset * sizeof(T), length * sizeof(T));
  }

  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const ReductionFunction<T>* fn_;

  size_t chunks_;
  size_t chunkSize_;
  size_t chunkBytes_;

  T* inbox_[2];
  std::unique_ptr<transport::Buffer> sendDataBuf_[2];
  std::unique_ptr<transport::Buffer> recvDataBuf_[2];

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace gloo
