#include "allreduce_grid_ft2.h"


namespace gloo {

template<typename T>
void AllreduceGridFT2<T>::setupProxy(AllreduceGridFT2::ProxyNode &proxy, grid::Node &original, bool requestNotification) {
  proxy.rank_ = original.rank_;

  for (int i = 0; i < groups_ - 1; i++) {
    proxy.ptrs_.push_back(static_cast<T *>(malloc(original.groupReduceNumElems_ * sizeof(T))));
  }

  std::vector<std::unique_ptr<transport::Buffer>> recvBackupBufs;
  std::vector<int> peerRanks = getCrossGroupPeers(original.rank_);

  for (int i = 0; i < groups_ - 1; i++) {
    auto &backupPair = context_->getPair(peerRanks[i]);
    auto slot = getSlot(myRank_, peerRanks[i], true);
    int recvSize = original.groupReduceNumElems_;

    recvBackupBufs.push_back( backupPair->createRecvBuffer(slot, proxy.ptrs_[i], recvSize * sizeof(T)) );
  }

  for (int peerRank : peerRanks) {
    auto& pair = this->getPair(peerRank);
    auto slot = getSlot(myRank_, peerRank);

    proxy.sendCrossGroupDataBufs_[peerRank] = pair->createSendBuffer(slot, ptrs_[0], bytes_);
    proxy.sendCrossGroupNotificationBufs_[peerRank] = pair->createSendBuffer(slot + 2, &dummy_, sizeof(dummy_));
    proxy.recvCrossGroupNotificationBufs_[peerRank] = pair->createRecvBuffer(slot + 2, &dummy_, sizeof(dummy_));

  }

  auto& rightPair = this->getPair(original.rightPeerRank_);
  auto rightPairSlot = getSlot(myRank_, original.rightPeerRank_, true);

  for (int i = 0; i < 2; i++) {
    proxy.sendRingDataBufs_.push_back(
        rightPair->createSendBuffer(rightPairSlot + i, proxy.ptrs_[0], original.groupReduceNumElems_ * sizeof(T)));
  }
  proxy.recvRingNotificationBuf_ = rightPair->createRecvBuffer(rightPairSlot + 2, &dummy_, sizeof(dummy_));

  for (int i = 0; i < groups_ - 1; i++) {
    recvBackupBufs[i]->waitRecv();
  }

  for (int i = 1; i < proxy.ptrs_.size(); i++) {
    fn_->call(proxy.ptrs_[0], proxy.ptrs_[i], original.groupReduceNumElems_);
  }

  for (int round = 2; round < round_; round++) {
    int chunkOffset, offset, length;
    std::tie(chunkOffset, offset, length) = getChunkPosPerRound(original.rank_, round);
    int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;
    if (length > 0) {
      fn_->call(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length);
    }
  }

  printElems(&proxy.ptrs_[0][0], 6);

  struct Status {
    Phase phase;
    int round;
    bool requestData;
    bool requestNotification;
  };

  Status myStatus {phase_, round_, false, requestNotification};
  Status peerStatus;

  auto sendStatusBuf =
      rightPair->createSendBuffer(requestSlotOffset_ + original.rightPeerRank_, &myStatus, sizeof(Status));
  auto recvStatusBuf =
      rightPair->createRecvBuffer(requestSlotOffset_ + original.rightPeerRank_, &peerStatus, sizeof(Status));

  sendStatusBuf->send();
  recvStatusBuf->waitRecv();

  std::cout << "peer status: " << peerStatus.round <<  (peerStatus.requestData ? " requested" : "") << std::endl;
  std::cout << "my status: " << myStatus.round << std::endl;


  // TODO: handle different phase
  proxy.startRound_ = peerStatus.round;


  for (int round = peerStatus.round ; round < round_ + 2; round++) {
    if (round == peerStatus.round && !peerStatus.requestData) {
      continue;
    }

    int chunkOffset, offset, length;
    std::tie(chunkOffset, offset, length) = getChunkPosPerRound(original.rightPeerRank_, round % chunks_);
    int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;

    std::cout << "send chunk " << chunkOffset << " ("  << proxy.ptrs_[0][localOffset] << ")" << std::endl;
    proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));
  }

  if (peerStatus.round < myStatus.round) {
    for (int i = peerStatus.round; i < myStatus.round; i++) {
      proxy.recvRingNotificationBuf_->waitRecv();
    }
  }


}


template<typename T>
void AllreduceGridFT2<T>::proxyInGroupReduceScatter() {
  ProxyNode& proxy = proxyNodes_[0];

  int chunkOffset, offset, length;
  std::tie(chunkOffset, offset, length) = getChunkPosPerRound(proxy.rank_, round_);

  int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;

  if (length > 0) {
    fn_->call(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length);
  }

  if (round_ >= proxy.startRound_) {
    std::cout << "proxy wait notification" << std::endl;
    proxy.recvRingNotificationBuf_->waitRecv();
  }

  std::cout << "proxy send data at " << chunkOffset << " ("  << proxy.ptrs_[0][localOffset] << ")" << std::endl;
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
    std::cout << "proxy wait notification" << std::endl;
    proxy.recvRingNotificationBuf_->waitRecv();

    std::cout << "proxy send data at " << chunkOffset << " ("  << proxy.ptrs_[0][localOffset] << ")" << std::endl;
    proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));
  }

  printElems(&proxy.ptrs_[0][0], 6);
}


template<typename T>
void AllreduceGridFT2<T>::proxyCrossGroupAllGather() {
  ProxyNode& proxy = proxyNodes_[0];

  for (auto peerRank : getCrossGroupPeers(proxy.rank_)) {
    int offset = allNodes_[myRank_].groupReduceOffset_;
    int length = allNodes_[myRank_].groupReduceNumElems_;
    proxy.sendCrossGroupDataBufs_[peerRank]->send(offset * sizeof(T), length * sizeof(T));
  }

  for (auto peerRank : getCrossGroupPeers(proxy.rank_)) {
    proxy.sendCrossGroupNotificationBufs_[peerRank]->send();
  }

  for (auto peerRank : getCrossGroupPeers(proxy.rank_)) {
    proxy.recvCrossGroupNotificationBufs_[peerRank]->waitRecv();
  }

}


template<typename T>
gloo::AllreduceGridFT2<T>::ProxyNode::ProxyNode() {
  rank_ = -1;
}


}
