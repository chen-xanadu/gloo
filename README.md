[Link to original README](https://github.com/chen-xanadu/gloo/blob/master/README_GLOO.md)

Added grid-based all-reduce implementation:
- [Fault-tolerant version](https://github.com/chen-xanadu/gloo/blob/master/gloo/allreduce_grid_ft.h)
- [Non fault-tolerant version](https://github.com/chen-xanadu/gloo/blob/master/gloo/allreduce_grid.h)


Example Script:
- [local_9.sh](https://github.com/chen-xanadu/gloo/blob/master/script/local_9.sh): launch 9 local processes, inject failure in rank-0 process
