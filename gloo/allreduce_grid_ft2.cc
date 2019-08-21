#include "allreduce_grid_ft2.h"


namespace gloo {

template<typename T>
void AllreduceGridFT2<T>::setupProxy(AllreduceGridFT2::ProxyNode &proxy, grid::Node &original, bool requestNotification) {
  proxy.rank_ = original.rank_;

  for (int i = 0; i < groups_ - 1; i++) {
    proxy.ptrs_.push_back(static_cast<T *>(malloc(original.groupReduceNumElems_ * sizeof(T))));
  }

  // Setup right pair buffers
  auto& rightPair = this->getPair(original.rightPeerRank_);
  auto rightPairSlot = getSlot(myRank_, original.rightPeerRank_, true);

  for (int i = 0; i < 2; i++) {
    proxy.sendRingDataBufs_.push_back(
        rightPair->createSendBuffer(rightPairSlot + i, proxy.ptrs_[0], original.groupReduceNumElems_ * sizeof(T)));
  }
  proxy.recvRingNotificationBuf_ = rightPair->createRecvBuffer(rightPairSlot + 2, &dummy_, sizeof(dummy_));

  Status myStatus {phase_, round_, false, requestNotification};
  Status peerStatus;

  auto sendStatusBuf =
      rightPair->createSendBuffer(requestSlotOffset_ + original.rightPeerRank_, &myStatus, sizeof(Status));
  auto recvStatusBuf =
      rightPair->createRecvBuffer(requestSlotOffset_ + original.rightPeerRank_, &peerStatus, sizeof(Status));

  DEBUG("exchange status");
  sendStatusBuf->send();
  recvStatusBuf->waitRecv();

  DEBUG("peer status: " << peerStatus.round <<  (peerStatus.requestData ? " requested" : ""));
  DEBUG("my status: " << myStatus.round);

  // TODO: handle different phase
  proxy.startRound_ = peerStatus.round;
  proxy.startPhase_ = peerStatus.phase;


  // Setup cross group buffers
  std::vector<int> peerRanks = getCrossGroupPeers(original.rank_);
  std::vector<std::unique_ptr<transport::UnboundBuffer>> notifs;
  std::vector<std::unique_ptr<transport::Buffer>> recvBackupBufs;

  for (int peerRank : peerRanks) {
    auto &pair = this->getPair(peerRank);
    auto slot = getSlot(myRank_, peerRank);

    proxy.sendCrossGroupDataBufs_[peerRank] = pair->createSendBuffer(slot, ptrs_[0], bytes_);
    proxy.recvCrossGroupNotificationBufs_[peerRank] = pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));
  }


  // Recv backup if necessary
  struct {
    int recoveryPeerRank;
    bool isRequested;
  } backupRequestMsg;

  backupRequestMsg.recoveryPeerRank = myRank_;
  backupRequestMsg.isRequested = peerStatus.phase < InGroupAllGather ||
      (peerStatus.phase == InGroupAllGather && peerStatus.round < 2);

  for (int dstRank : peerRanks) {
    notifs.push_back(context_->createUnboundBuffer(&backupRequestMsg, sizeof(backupRequestMsg)));
    notifs.back()->send(dstRank, requestSlotOffset_ + dstRank);
  }
  for (auto & notif : notifs) {
    notif->waitSend();
  }

  if (backupRequestMsg.isRequested) {
    for (int i = 0; i < groups_ - 1; i++) {
      auto &backupPair = context_->getPair(peerRanks[i]);
      auto slot = getSlot(myRank_, peerRanks[i], true);
      int recvSize = original.groupReduceNumElems_;

      recvBackupBufs.push_back(backupPair->createRecvBuffer(slot, proxy.ptrs_[i], recvSize * sizeof(T)));
    }

    for (int i = 0; i < groups_ - 1; i++) {
      recvBackupBufs[i]->waitRecv();
    }

    for (int i = 1; i < proxy.ptrs_.size(); i++) {
      fn_->call(proxy.ptrs_[0], proxy.ptrs_[i], original.groupReduceNumElems_);
    }

  }

  // Re-construct previous rounds
  int stopRound = phase_ <= InGroupReduceScatter ? round_ : chunks_;
  for (int round = 2; round < stopRound; round++) {
    int chunkOffset, offset, length;
    std::tie(chunkOffset, offset, length) = getChunkPosPerRound(original.rank_, round);
    int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;
    if (length > 0) {
      fn_->call(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length);
    }
  }

  stopRound = phase_ >= InGroupAllGather ? round_ : 0;
  for (int round = 0; round < stopRound; round++) {
    int chunkOffset, offset, length;
    std::tie(chunkOffset, offset, length) = getChunkPosPerRound(original.rank_, round);
    int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;
    if (length > 0) {
      memcpy(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length * sizeof(T));
    }
  }


  if (peerStatus.phase <= InGroupReduceScatter && phase_ <= InGroupReduceScatter) {
    stopRound = round_ + 2;
  } else if (peerStatus.phase >= InGroupAllGather && phase_ >= InGroupAllGather) {
    stopRound = round_ + 2;
  } else if (peerStatus.phase <= InGroupReduceScatter && phase_ >= InGroupAllGather) {
    stopRound = round_ + chunks_ + 2;
  } else {
    stopRound = 2;
  }

  for (int round = peerStatus.round ; round < stopRound; round++) {
    if (round == peerStatus.round && !peerStatus.requestData) {
      continue;
    }

    int chunkOffset, offset, length;
    std::tie(chunkOffset, offset, length) = getChunkPosPerRound(original.rightPeerRank_, round % chunks_);
    int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;

    DEBUG("send chunk " << chunkOffset << " ("  << proxy.ptrs_[0][localOffset] << ")");
    proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));
  }

  if (peerStatus.round < myStatus.round) {
    for (int i = peerStatus.round; i < myStatus.round; i++) {
      proxy.recvRingNotificationBuf_->waitRecv();
    }
  }

  if (backupRequestMsg.isRequested) {
    notifs.clear();
    for (int dstRank : peerRanks) {
      notifs.push_back(context_->createUnboundBuffer(&backupRequestMsg, sizeof(backupRequestMsg)));
      notifs[notifs.size() - 1]->send(dstRank, requestSlotOffset_ + dstRank);
    }
    for (auto &notif : notifs) {
      notif->waitSend();
    }
  }

}


template<typename T>
void AllreduceGridFT2<T>::repairRightPeer(bool requestNotification) {
  DEBUG("repair right");

  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[myNode.rightPeerRank_];
  int failedNodeRank = failedNode.rank_;

//  std::thread signal(&AllreduceGridFT2::signalNodeFailure, this, failedNodeRank);
  signalNodeFailure(failedNodeRank);

  proxyNodes_.emplace_back();

  ProxyNode& proxy = proxyNodes_.back();

  DEBUG("proxy");
  setupProxy(proxy, failedNode, requestNotification);


//  signal.join();

  DEBUG("finished");

}


template<typename T>
void AllreduceGridFT2<T>::repairLeftPeer() {
  DEBUG("repair left");
  std::unique_lock<std::mutex> lock(leftPairMutex_);

  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[myNode.leftPeerRank_];

  auto& leftPair = this->getPair(failedNode.leftPeerRank_);
  auto leftPairSlot = getSlot(myRank_, failedNode.leftPeerRank_, true);

  ringInbox_.clear();
  recvRingDataBufs_.clear();
  for (int i = 0; i < 2; i++) {
    int recvSize = myNode.chunkSize_;
    ringInbox_.emplace_back(recvSize);
    recvRingDataBufs_.push_back(
        leftPair->createRecvBuffer(leftPairSlot + i, &ringInbox_[i][0], recvSize * sizeof(T)));
  }
  sendRingNotificationBuf_ = leftPair->createSendBuffer(leftPairSlot + 2, &dummy_, sizeof(dummy_));

  int round = (phase_ <= InGroupReduceScatter && round_ < 2) ? 2 : round_;
  Status myStatus {phase_, round, requestRoundData_, false};
  Status peerStatus;

  auto sendStatusBuf =
      leftPair->createSendBuffer(requestSlotOffset_ + myRank_, &myStatus, sizeof(Status));
  auto recvStatusBuf =
      leftPair->createRecvBuffer(requestSlotOffset_ + myRank_, &peerStatus, sizeof(Status));

  DEBUG("exchange status");
  sendStatusBuf->send();
  recvStatusBuf->waitRecv();

  DEBUG("peer status: " << peerStatus.round <<  (peerStatus.requestNotification ? " requested" : ""));
  DEBUG("my status: " << myStatus.round);

  // TODO: handle different phase
  if (roundNotificationSent_) {
    sendRingNotificationBuf_->send();
  }


  lock.unlock();
  leftPairCV_.notify_one();

  DEBUG("finish");
}


template<typename T>
void AllreduceGridFT2<T>::repairGroupPeer(int failedNodeRank) {
  DEBUG("repair group peer");
  std::unique_lock<std::mutex> lock(proxyMutex_);
  proxyCV_.wait(lock, [&]{ return phase_ == Completed;});

  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[failedNodeRank];

  std::vector<int> peerRanks = getCrossGroupPeers(failedNode.rank_);
  std::unordered_map<int, std::unique_ptr<transport::UnboundBuffer>> recvRequestBufs;
  std::unordered_map<int, AllGatherRequest> requests;

  for (int peerRank : peerRanks) {
    auto& pair = this->getPair(peerRank);
    auto slot = getSlot(myRank_, peerRank);

    requests[peerRank] = {-1, -1, -1};
    recvRequestBufs[peerRank] =
        context_->createUnboundBuffer(&requests[peerRank], sizeof(AllGatherRequest));

    sendCrossGroupDataBufs_[peerRank] = pair->createSendBuffer(slot, ptrs_[0], bytes_);
    recvCrossGroupNotificationBufs_[peerRank] = pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));

    recvRequestBufs[peerRank]->recv(peerRank, slot + 1);
    DEBUG("wait requests from  " << peerRank << " (slot " << slot + 1 << ")");
  }


  for (auto peerRank : peerRanks) {
    recvRequestBufs[peerRank]->waitRecv();
    int offset = requests[peerRank].offset;
    int length = requests[peerRank].length;
    DEBUG("send data to " << peerRank << " (slot " << sendCrossGroupDataBufs_[peerRank]->getSlot() << ")");
    sendCrossGroupDataBufs_[peerRank]->send(offset * sizeof(T), length * sizeof(T));
  }

  for (auto peerRank : peerRanks) {
    recvCrossGroupNotificationBufs_[peerRank]->waitRecv();
  }

  DEBUG("finish group peer");
}


template<typename T>
void AllreduceGridFT2<T>::repairCrossGroupPeer(int failedNodeRank) {
  DEBUG("repair cross group peer");
  std::unique_lock<std::mutex> lock(crossGroupMutex_);
  failedNodeRanks_.insert(failedNodeRank);

  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[failedNodeRank];

  struct {
    int recoveryPeerRank;
    bool isRequested;
  } backupRequestMsg;

  std::unique_ptr<transport::UnboundBuffer> repairNotificationBuf =
      context_->createUnboundBuffer(&backupRequestMsg, sizeof(backupRequestMsg));


  myNode.crossGroupPeerRanks_.erase(failedNode.group_);

  std::vector<T> inbox(failedNode.groupReduceNumElems_);
  std::unordered_map<int, std::unique_ptr<transport::UnboundBuffer>> sendRequestBufs;


  lock.unlock();
  crossGroupCV_.notify_one();

  repairNotificationBuf->recv(failedNode.groupPeerRanks_, requestSlotOffset_ + myRank_);
  repairNotificationBuf->waitRecv();

  if (backupRequestMsg.isRequested) {
    int slot = getSlot(myRank_, backupRequestMsg.recoveryPeerRank, true);
    auto& pair = this->getPair(backupRequestMsg.recoveryPeerRank);
    auto sendBackupBuf = pair->createSendBuffer(slot, ptrs_[0], bytes_);

    int offset = failedNode.groupReduceOffset_;
    int length = failedNode.groupReduceNumElems_;
    sendBackupBuf->send(offset * sizeof(T), length * sizeof(T));

    repairNotificationBuf->recv(backupRequestMsg.recoveryPeerRank, requestSlotOffset_ + myRank_);
    repairNotificationBuf->waitRecv();
  }

  // Recv all-gather data
  std::vector<int> failedNodeGroupPeerRanks = getGroupPeers(failedNodeRank);
  std::unordered_map<int, AllGatherRequest> requests;

  int actualGroupSize = getActualGroupSize(failedNodeRank);
  for (int peerRank : failedNodeGroupPeerRanks) {
    int actualGroupRank = getActualGroupRank(peerRank);
    int length = allNodes_[peerRank].groupReduceNumElems_ / actualGroupSize;
    int offset = allNodes_[peerRank].groupReduceOffset_ + length * actualGroupRank;
    if (actualGroupRank == actualGroupSize - 1) {
      length = allNodes_[peerRank].groupReduceOffset_ + allNodes_[peerRank].groupReduceNumElems_ - offset;
    }

    requests[peerRank] = {myRank_, offset, length};
    sendRequestBufs[peerRank] =
        context_->createUnboundBuffer(&requests[peerRank], sizeof(AllGatherRequest));

    int slot = getSlot(myRank_, peerRank);
    auto& pair = this->getPair(peerRank);

    sendRequestBufs[peerRank]->send(peerRank, slot + 1);
    DEBUG("send request to  " << peerRank << " (offset: " << offset << ", length: " << length << ")");

    int localOffset = offset - allNodes_[peerRank].groupReduceOffset_;
    recvCrossGroupDataBufs_[peerRank] = pair->createRecvBuffer(slot, &inbox[localOffset], length * sizeof(T));
    sendCrossGroupNotificationBufs_[peerRank] = pair->createSendBuffer(slot + 2, &dummy_, sizeof(dummy_));
  }

  for (int peerRank : failedNodeGroupPeerRanks) {
    DEBUG("wait data from " << peerRank << " (slot " << recvCrossGroupDataBufs_[peerRank]->getSlot() << ")");
    recvCrossGroupDataBufs_[peerRank]->waitRecv();
  }

  printElems(&inbox[0], failedNode.groupReduceNumElems_);

  for (int peerRank : failedNodeGroupPeerRanks) {
    sendCrossGroupNotificationBufs_[peerRank]->send();
  }

  int offset = failedNode.groupReduceOffset_;
  int length = failedNode.groupReduceNumElems_;
  if (length > 0) {
    memcpy(&ptrs_[0][offset], &inbox[0], length * sizeof(T));
  }

  DEBUG("finish cross group peer");
}


template<typename T>
void AllreduceGridFT2<T>::proxyInGroupReduceScatter() {
  ProxyNode& proxy = proxyNodes_[0];

  int chunkOffset, offset, length;
  std::tie(chunkOffset, offset, length) = getChunkPosPerRound(proxy.rank_, round_);

  int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;

  if (length > 0 && round_ >= 2) {
    fn_->call(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length);
  }

  if (proxy.startPhase_ <= InGroupReduceScatter && round_ >= std::max(proxy.startRound_, 2)) {
    DEBUG("proxy wait notification");
    proxy.recvRingNotificationBuf_->waitRecv();
  }

  DEBUG("proxy send data at " << chunkOffset << " ("  << proxy.ptrs_[0][localOffset] << ")");
  proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));

  printElems(&proxy.ptrs_[0][0], 6);
}


template<typename T>
void AllreduceGridFT2<T>::proxyInGroupAllGather() {
  ProxyNode& proxy = proxyNodes_[0];

  int chunkOffset, offset, length;
  std::tie(chunkOffset, offset, length) = getChunkPosPerRound(proxy.rank_, round_);

  int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;

  if (length > 0) {
    memcpy(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length * sizeof(T));
  }

  if (round_ < (chunks_ - 4)) {
    if (proxy.startPhase_ < InGroupAllGather ||
        (proxy.startPhase_ == InGroupAllGather && round_ >= proxy.startRound_)) {
      DEBUG("proxy wait notification");
      proxy.recvRingNotificationBuf_->waitRecv();
    }

    DEBUG("proxy send data at " << chunkOffset << " ("  << proxy.ptrs_[0][localOffset] << ")");
    proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));
  }

  printElems(&proxy.ptrs_[0][0], 6);
}


template<typename T>
void AllreduceGridFT2<T>::proxyCrossGroupAllGather() {
  ProxyNode& proxy = proxyNodes_[0];

  std::vector<int> peerRanks = getCrossGroupPeers(proxy.rank_);
  std::unordered_map<int, std::unique_ptr<transport::UnboundBuffer>> recvRequestBufs;
  std::unordered_map<int, AllGatherRequest> requests;

  for (int peerRank : peerRanks) {
    auto& pair = this->getPair(peerRank);
    auto slot = getSlot(myRank_, peerRank);

    requests[peerRank] = {-1, -1, -1};
    recvRequestBufs[peerRank] =
        context_->createUnboundBuffer(&requests[peerRank], sizeof(AllGatherRequest));

    recvRequestBufs[peerRank]->recv(peerRank, slot + 1);
    DEBUG("wait requests from  " << peerRank << " (slot " << slot + 1 << ")");
  }


  for (auto peerRank : getCrossGroupPeers(proxy.rank_)) {
    recvRequestBufs[peerRank]->waitRecv();
    int offset = requests[peerRank].offset;
    int length = requests[peerRank].length;
    DEBUG("send data to " << peerRank << " (slot " << proxy.sendCrossGroupDataBufs_[peerRank]->getSlot() << ")");
    proxy.sendCrossGroupDataBufs_[peerRank]->send(offset * sizeof(T), length * sizeof(T));
  }

  for (auto peerRank : getCrossGroupPeers(proxy.rank_)) {
    DEBUG("wait notifs from  " << peerRank << " (slot " << proxy.recvCrossGroupNotificationBufs_[peerRank]->getSlot() << ")");
    proxy.recvCrossGroupNotificationBufs_[peerRank]->waitRecv();
  }

}


template<typename T>
void AllreduceGridFT2<T>::repairRightPeerOnStartup() {
  DEBUG("repair right on startup");

  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[myNode.rightPeerRank_];
  int failedNodeRank = failedNode.rank_;

  proxyNodes_.emplace_back();

  ProxyNode& proxy = proxyNodes_.back();

  proxy.rank_ = failedNode.rank_;

  for (int i = 0; i < groups_ - 1; i++) {
    proxy.ptrs_.push_back(static_cast<T *>(malloc(failedNode.groupReduceNumElems_ * sizeof(T))));
  }

  // Setup right pair buffers
  auto& rightPair = this->getPair(failedNode.rightPeerRank_);
  auto rightPairSlot = getSlot(myRank_, failedNode.rightPeerRank_, true);

  for (int i = 0; i < 2; i++) {
    proxy.sendRingDataBufs_.push_back(
        rightPair->createSendBuffer(rightPairSlot + i, proxy.ptrs_[0], failedNode.groupReduceNumElems_ * sizeof(T)));
  }
  proxy.recvRingNotificationBuf_ = rightPair->createRecvBuffer(rightPairSlot + 2, &dummy_, sizeof(dummy_));


  // Setup cross group buffers
  std::vector<int> peerRanks = getCrossGroupPeers(failedNode.rank_);
  std::vector<std::unique_ptr<transport::Buffer>> recvBackupBufs;
  std::vector<std::unique_ptr<transport::UnboundBuffer>> notifs;


  for (int peerRank : peerRanks) {
    auto &pair = this->getPair(peerRank);
    auto slot = getSlot(myRank_, peerRank);

    proxy.sendCrossGroupDataBufs_[peerRank] = pair->createSendBuffer(slot, ptrs_[0], bytes_);
    proxy.recvCrossGroupNotificationBufs_[peerRank] = pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));
  }


  // Recv backup
  for (int i = 0; i < groups_ - 1; i++) {
    auto &backupPair = context_->getPair(peerRanks[i]);
    auto slot = getSlot(myRank_, peerRanks[i], true);
    int recvSize = failedNode.groupReduceNumElems_;

    recvBackupBufs.push_back(backupPair->createRecvBuffer(slot, proxy.ptrs_[i], recvSize * sizeof(T)));
  }

  for (int i = 0; i < groups_ - 1; i++) {
    recvBackupBufs[i]->waitRecv();
  }

  printElems(&proxy.ptrs_[0][0], 6);


  for (int i = 1; i < proxy.ptrs_.size(); i++) {
    fn_->call(proxy.ptrs_[0], proxy.ptrs_[i], failedNode.groupReduceNumElems_);
  }

  for (int dstRank : peerRanks) {
    notifs.push_back(context_->createUnboundBuffer(&dummy_, sizeof(dummy_)));
    notifs.back()->send(dstRank, requestSlotOffset_ + dstRank);
  }
  for (auto & notif : notifs) {
    notif->waitSend();
  }


  DEBUG("finished");
}


template<typename T>
void AllreduceGridFT2<T>::repairLeftPeerOnStartup() {
  DEBUG("repair left on startup");
  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[myNode.leftPeerRank_];

  auto& leftPair = this->getPair(failedNode.leftPeerRank_);
  auto leftPairSlot = getSlot(myRank_, failedNode.leftPeerRank_, true);

  ringInbox_.clear();
  recvRingDataBufs_.clear();
  for (int i = 0; i < 2; i++) {
    int recvSize = myNode.chunkSize_;
    ringInbox_.emplace_back(recvSize);
    recvRingDataBufs_.push_back(
        leftPair->createRecvBuffer(leftPairSlot + i, &ringInbox_[i][0], recvSize * sizeof(T)));
  }
  sendRingNotificationBuf_ = leftPair->createSendBuffer(leftPairSlot + 2, &dummy_, sizeof(dummy_));

  DEBUG("finish");
}


template<typename T>
void AllreduceGridFT2<T>::repairCrossGroupPeerOnStartup(int failedNodeRank) {
  DEBUG("repair cross group peer on startup");
  std::unique_lock<std::mutex> lock(crossGroupMutex_);
  failedNodeRanks_.insert(failedNodeRank);

  grid::Node& myNode = allNodes_[myRank_];
  grid::Node& failedNode = allNodes_[failedNodeRank];


  myNode.crossGroupPeerRanks_.erase(failedNode.group_);

  std::vector<T> inbox(failedNode.groupReduceNumElems_);
  std::unordered_map<int, std::unique_ptr<transport::UnboundBuffer>> sendRequestBufs;
  std::unique_ptr<transport::UnboundBuffer> repairNotificationBuf =
      context_->createUnboundBuffer(&dummy_, sizeof(dummy_));


  int slot = getSlot(myRank_, failedNode.leftPeerRank_, true);
  auto& pair = this->getPair(failedNode.leftPeerRank_);
  auto sendBackupBuf = pair->createSendBuffer(slot, ptrs_[0], bytes_);

  int offset = failedNode.groupReduceOffset_;
  int length = failedNode.groupReduceNumElems_;
  sendBackupBuf->send(offset * sizeof(T), length * sizeof(T));

  printElems(&ptrs_[0][offset], length);


  repairNotificationBuf->recv(failedNode.groupPeerRanks_, requestSlotOffset_ + myRank_);
  repairNotificationBuf->waitRecv();


  lock.unlock();
  crossGroupCV_.notify_one();

  // Recv all-gather data
  std::vector<int> failedNodeGroupPeerRanks = getGroupPeers(failedNodeRank);
  std::unordered_map<int, AllGatherRequest> requests;

  int actualGroupSize = getActualGroupSize(failedNodeRank);
  for (int peerRank : failedNodeGroupPeerRanks) {
    int actualGroupRank = getActualGroupRank(peerRank);
    int length = allNodes_[peerRank].groupReduceNumElems_ / actualGroupSize;
    int offset = allNodes_[peerRank].groupReduceOffset_ + length * actualGroupRank;
    if (actualGroupRank == actualGroupSize - 1) {
      length = allNodes_[peerRank].groupReduceOffset_ + allNodes_[peerRank].groupReduceNumElems_ - offset;
    }

    requests[peerRank] = {myRank_, offset, length};
    sendRequestBufs[peerRank] =
        context_->createUnboundBuffer(&requests[peerRank], sizeof(AllGatherRequest));

    int slot = getSlot(myRank_, peerRank);
    auto& pair = this->getPair(peerRank);

    sendRequestBufs[peerRank]->send(peerRank, slot + 1);
    DEBUG("send request to  " << peerRank << " (offset: " << offset << ", length: " << length << ")");

    int localOffset = offset - allNodes_[peerRank].groupReduceOffset_;
    recvCrossGroupDataBufs_[peerRank] = pair->createRecvBuffer(slot, &inbox[localOffset], length * sizeof(T));
    sendCrossGroupNotificationBufs_[peerRank] = pair->createSendBuffer(slot + 2, &dummy_, sizeof(dummy_));
  }

  for (int peerRank : failedNodeGroupPeerRanks) {
    DEBUG("wait data from " << peerRank << " (slot " << recvCrossGroupDataBufs_[peerRank]->getSlot() << ")");
    recvCrossGroupDataBufs_[peerRank]->waitRecv();
  }

  printElems(&inbox[0], failedNode.groupReduceNumElems_);

  for (int peerRank : failedNodeGroupPeerRanks) {
    sendCrossGroupNotificationBufs_[peerRank]->send();
  }

  offset = failedNode.groupReduceOffset_;
  length = failedNode.groupReduceNumElems_;
  if (length > 0) {
    memcpy(&ptrs_[0][offset], &inbox[0], length * sizeof(T));
  }

  DEBUG("finish cross group peer");
}

template<typename T>
gloo::AllreduceGridFT2<T>::ProxyNode::ProxyNode() {
  rank_ = -1;
}


}
