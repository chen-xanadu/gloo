set(GLOO_RENDEZVOUS_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/file_store.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/hash_store.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/prefix_store.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/store.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/redis_store.h"
  )

set(GLOO_RENDEZVOUS_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/file_store.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/hash_store.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/prefix_store.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/store.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/context.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/redis_store.cc"
  )

if(USE_REDIS)
  list(APPEND GLOO_RENDEZVOUS_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/redis_store.cc"
    )
  list(APPEND GLOO_RENDEZVOUS_HDRS
    "${CMAKE_CURRENT_SOURCE_DIR}/redis_store.h"
    )
endif()

list(APPEND GLOO_SRCS ${GLOO_RENDEZVOUS_SRCS})
list(APPEND GLOO_HDRS ${GLOO_RENDEZVOUS_HDRS})
set(GLOO_SRCS ${GLOO_SRCS} PARENT_SCOPE)
set(GLOO_HDRS ${GLOO_HDRS} PARENT_SCOPE)
