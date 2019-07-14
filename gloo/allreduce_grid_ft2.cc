#include "allreduce_grid_ft2.h"


namespace gloo {

template<typename T>
void AllreduceGridFT2<T>::setupProxy(AllreduceGridFT2::ProxyNode &proxy, grid::Node &original) {
  proxy.rank_ = original.rank_;

  for (int i = 0; i < groups_ - 1; i++) {
    proxy.ptrs_.push_back(static_cast<T *>(malloc(original.groupReduceNumElems_ * sizeof(T))));
  }

  std::vector<std::unique_ptr<transport::Buffer>> recvBackupBufs;
  std::vector<int> peerRanks = getCrossGroupPeers(original.rank_);

  recvBackupBufs.reserve(peerRanks.size());
  for (int i = 0; i < groups_ - 1; i++) {
    auto &backupPair = context_->getPair(peerRanks[i]);
    auto slot = getSlot(myRank_, peerRanks[i], true);
    int recvSize = original.groupReduceNumElems_;

    recvBackupBufs[i] = backupPair->createRecvBuffer(slot, proxy.ptrs_[i], recvSize * sizeof(T));
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

  int chunkOffset, offset, length;
  std::tie(chunkOffset, offset, length) = getChunkPosPerRound(proxy.rank_, round_-2);
  int localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;
  if (length > 0) {
    fn_->call(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length);
  }
  proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));


  std::tie(chunkOffset, offset, length) = getChunkPosPerRound(proxy.rank_, round_-1);
  localOffset = offset - allNodes_[proxy.rank_].groupReduceOffset_;
  if (length > 0) {
    fn_->call(&proxy.ptrs_[0][localOffset], &ptrs_[0][offset], length);
  }
  proxy.sendRingDataBufs_[chunkOffset & 1]->send(localOffset * sizeof(T), length * sizeof(T));


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

  proxy.recvRingNotificationBuf_->waitRecv();

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
    proxy.recvRingNotificationBuf_->waitRecv();

    // Copy accumulated chunks
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
