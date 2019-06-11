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
    groupSize_ = contextSize_ / 2;
    group_ = contextRank_ / groupSize_;
    groupRank_ = contextRank_ - group_ * groupSize_;

    copyPairRank_ = (contextRank_ + groupSize_) % contextSize_;
    copyPairFailed_ = false;
    leftPairRank_ = (groupSize_ + groupRank_ - 1) % groupSize_ + group_ * groupSize_;
    rightPairRank_ = (groupRank_ + 1) % groupSize_ + group_ * groupSize_;
    rightPairFailed_ = false;

    // Use chunks of no less than 1024 bytes (256 * sizeof(float))
    constexpr unsigned long minSize = 256;
    chunks_ = this->groupSize_ * 2;
    chunkSize_ = std::max(minSize, (count_ + chunks_ - 1) / chunks_);
    chunkBytes_ = chunkSize_ * sizeof(T);

    // Allocate inboxes
    for (int i = 0; i < 3; i++) {
      inbox_[i] = static_cast<T*>(malloc(bytes_));
    }

    if (this->contextSize_ == 1) {
      return;
    }

    auto& copyPair = this->getPair(copyPairRank_);
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
    auto slot = this->context_->nextSlot();
    sendDataBuf_[2] =
        copyPair->createSendBuffer(slot, ptrs_[0], bytes_);
    recvDataBuf_[2] =
        copyPair->createRecvBuffer(slot, inbox_[2], bytes_);

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
    auto copyNotificationSlot = this->context_->nextSlot();
    copySendNotificationBuf_ =
        copyPair->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    copyRecvNotificationBuf_ =
        copyPair->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));

  }

  virtual ~AllreduceRingChunked() {
    for (int i = 0; i < 3; i++) {
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

//    std::cout << "---- Starting initial pair copying ----" << std::endl;
    if (!copyPairFailed_) {
      sendDataBuf_[2]->trySend();
      recvDataBuf_[2]->tryWaitRecv();
      fn_->call(ptrs_[0], inbox_[2], count_);

      copyHeartbeat(-1);
    }


    std::cout << "---- Starting in-group allreduce phase 1 ----" << std::endl;
    // Kick off copying initial chunks
    copyChunkAtOffset(2 * this->groupRank_);
    copyChunkAtOffset(2 * this->groupRank_ + 1);

    if (copyPairFailed_) {
      copyChunkAtOffset2(2 * this->groupRank_);
      copyChunkAtOffset2(2 * this->groupRank_ + 1);
    }

    if (contextRank_ == 1) {
      exit(0);
    }

    // Start with reduction of previously copied chunk
    for (int round = 2; round < chunks_; round++) {


//      std::cout << "-- Starting phase 1 round " << round << " --" << std::endl;
//      auto start = std::chrono::high_resolution_clock::now();

      // We loop over all chunks starting at 2, since we just sent two
      // chunks to fill both buffers. Imagine a square grid with
      // chunks of memory laid out vertically and nodes horizontally.
      // The diagonal of this grid marks which nodes sends which
      // chunks of memory in the prelude. Processing happens by moving
      // this diagonal forward and have it wrap around the edge. This
      // means that node with rank 0 at round 2 will process the last
      // chunk. This explains why we subtract the round in the offset
      // equation below.
      //
      // Because we're dealing with double buffering in this
      // implementation, we have twice the number of chunks and
      // process them in pairs. This explains why we ignore the LSB on
      // the round number when subtracting it. The LSB is later added
      // to flip back and forth between the two buffers for this pair
      // of chunks. The number of chunks is finally added to make sure
      // we can wrap correctly (no modulo against negative number).
      //
      auto chunkOffset = ((2 * this->groupRank_) - (round & ~0x1) +
                          (round & 0x1) + chunks_) %
                         chunks_;
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

//      std::cout << "Receiving data from " << leftPairRank_ << std::endl;
      // Wait for inbox write to complete
//      recvDataBuf_[chunkOffset & 1]->waitRecv();
      recvData(chunkOffset & 1);

      // Reduce
      if (length > 0) {
        fn_->call(&ptrs_[0][offset], inbox_[chunkOffset & 1], length);
      }

//      std::cout << "Sending notification to " << leftPairRank_ << std::endl;
      // Send notification to node on the left that
      // this node is ready for an inbox write.
//      sendNotificationBuf_->send();
      sendNotification();

      if (!copyPairFailed_) {
        copyHeartbeat(chunkOffset);
      }


//      std::cout << "Waiting notification from " << rightPairRank_ << std::endl;
      // Wait for notification from node on the right
      // to be sure this node can start an inbox write.
//      recvNotificationBuf_->waitRecv();
      if (!rightPairFailed_) {
        recvNotificationBuf_->tryWaitRecv();
      }

      if (copyPairFailed_) {
        recvNotificationBuf2_->tryWaitRecv();
      }

      if (!copyPairFailed_) {
        copyHeartbeat(chunkOffset);
      }


      // Copy accumulated chunk
      copyChunkAtOffset(chunkOffset);

      if (copyPairFailed_) {
        copyChunkAtOffset2(chunkOffset);
      }


//      copySendNotificationBuf_->send();
//      copyRecvNotificationBuf_->waitRecv();


//      auto end = std::chrono::high_resolution_clock::now();
//      std::cout << "Round " << round << " : "
//                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
//                << " us" << std::endl;
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
      recvDataBuf_[chunkOffset & 1]->tryWaitRecv();

      // Copy
      if (length > 0) {
        memcpy(&ptrs_[0][offset], inbox_[chunkOffset & 1], length * sizeof(T));
      }

      // Skip copying in the last two rounds
      if (round < (chunks_ - 4)) {
        // Send notification to node on the left that
        // this node is ready for an inbox write.
        sendNotificationBuf_->trySend();

        // Wait for notification from node on the right
        // to be sure this node can start an inbox write.
        if (!rightPairFailed_) {
          recvNotificationBuf_->tryWaitRecv();
        }
        if (copyPairFailed_) {
          recvNotificationBuf2_->tryWaitRecv();
        }

        // Copy accumulated chunks
        copyChunkAtOffset(chunkOffset);
        if (copyPairFailed_) {
          copyChunkAtOffset2(chunkOffset);
        }
      }

    }

    // Final barrier to make sure every node has finished
    // Otherwise, a second all reduce call might interfere
    // with one that it still in progress on some nodes.
    sendNotificationBuf_->trySend();
    if (!rightPairFailed_) {
      recvNotificationBuf_->tryWaitRecv();
    }
    if (copyPairFailed_) {
      recvNotificationBuf2_->tryWaitRecv();
    }

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }

 protected:

  void sendData(int chunkOffset, size_t offset, size_t length) {
    auto success = sendDataBuf_[chunkOffset & 0x1]->trySend(offset, length);
    if (!success) {
      std::cout << "sending data to " << rightPairRank_ <<  " failed" << std::endl;
      rightPairRank_ = (rightPairRank_ + groupSize_) % contextSize_;

      auto& rightPair = this->getPair(rightPairRank_);

      for (int i = 0; i < 2; i++) {
        auto slot = this->context_->nextSlot();
        sendDataBuf_[i] =
            rightPair->createSendBuffer(slot, ptrs_[0], bytes_);
      }
      auto notificationSlot = this->context_->nextSlot();
      recvNotificationBuf_ =
          rightPair->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));

      copyChunkAtOffset((chunkOffset + chunks_ - 1) % chunks_);
      copyChunkAtOffset(chunkOffset);
    }
  }

  void sendNotification() {
    auto success = sendNotificationBuf_->trySend();
    if (!success) {
      std::cout << "sending notification to " << leftPairRank_ <<  " failed" << std::endl;
      leftPairRank_ = (leftPairRank_ + groupSize_) % contextSize_;

      auto& leftPair = this->getPair(leftPairRank_);
      for (int i = 0; i < 2; i++) {
        free(inbox_[i]);
        inbox_[i] = static_cast<T*>(malloc(bytes_));
        auto slot = this->context_->nextSlot();
        recvDataBuf_[i] =
            leftPair->createRecvBuffer(slot, inbox_[i], chunkBytes_);
      }
      auto notificationSlot = this->context_->nextSlot();
      sendNotificationBuf_ =
          leftPair->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));

      sendNotification();
    }
  }

  void recvData(int bufIdx) {
    auto success = recvDataBuf_[bufIdx]->tryWaitRecv();

    if (!success) {
      std::cout << "recving data from " << leftPairRank_ <<  " failed" << std::endl;
      leftPairRank_ = (leftPairRank_ + groupSize_) % contextSize_;

      auto& leftPair = this->getPair(leftPairRank_);
      for (int i = 0; i < 2; i++) {
        auto slot = this->context_->nextSlot();
        free(inbox_[i]);
        inbox_[i] = static_cast<T*>(malloc(bytes_));
        recvDataBuf_[i] =
            leftPair->createRecvBuffer(slot, inbox_[i], chunkBytes_);
      }
      auto notificationSlot = this->context_->nextSlot();
      sendNotificationBuf_ =
          leftPair->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));

      recvData(bufIdx);
    }
  }

  void copyHeartbeat(int chunkOffset) {
    auto success = copySendNotificationBuf_->trySend() && copyRecvNotificationBuf_->tryWaitRecv();
    if (!success) {
      std::cout << "copy pair " << copyPairRank_ <<  " failed" << std::endl;
      copyPairFailed_ = true;
      int initialRightPairRank = (groupRank_ + 1) % groupSize_ + group_ * groupSize_;
      copyRightPairRank_ = (initialRightPairRank + groupSize_) % contextSize_;

      auto& copyRightPair = this->getPair(copyRightPairRank_);

      for (int i = 0; i < 2; i++) {
        auto slot = this->context_->nextSlot();
        sendDataBuf2_[i] =
            copyRightPair->createSendBuffer(slot, ptrs_[0], bytes_);
      }
      auto notificationSlot = this->context_->nextSlot();
      recvNotificationBuf2_ =
          copyRightPair->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));

      if (chunkOffset >= 0) {
        copyChunkAtOffset2((chunkOffset + chunks_ - 2) % chunks_);
        copyChunkAtOffset2((chunkOffset + chunks_ - 1) % chunks_);
      }
    }
  }



  void copyChunkAtOffset(int chunkOffset) {
    if (rightPairFailed_) {
      return;
    }
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
    bool success = sendDataBuf_[chunkOffset & 0x1]->trySend(
        offset * sizeof(T), length * sizeof(T));
    if (success) {
//      std::cout << "sent data to " << rightPairRank_ << std::endl;
    } else {
      rightPairFailed_ = true;
    }

  }

  void copyChunkAtOffset2(int chunkOffset) {
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
    bool success = sendDataBuf2_[chunkOffset & 0x1]->trySend(
        offset * sizeof(T), length * sizeof(T));
    if (success) {
//      std::cout << "sent extra data to " << copyRightPairRank_ << std::endl;
    }
  }

  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const ReductionFunction<T>* fn_;

  size_t chunks_;
  size_t chunkSize_;
  size_t chunkBytes_;

  int leftPairRank_;
  int rightPairRank_;
  int copyPairRank_;
  int groupRank_;
  int groupSize_;
  int group_;

  bool copyPairFailed_;
  int copyRightPairRank_;
  bool rightPairFailed_;

  T* inbox_[3];
  std::unique_ptr<transport::Buffer> sendDataBuf_[3];
  std::unique_ptr<transport::Buffer> recvDataBuf_[3];

  std::unique_ptr<transport::Buffer> sendDataBuf2_[2];
  std::unique_ptr<transport::Buffer> recvNotificationBuf2_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;

  std::unique_ptr<transport::Buffer> copySendNotificationBuf_;
  std::unique_ptr<transport::Buffer> copyRecvNotificationBuf_;
};

} // namespace gloo
