add_executable(example1 example1.cc)
add_executable(example2 example2.cc)
target_link_libraries(example1 gloo)
target_link_libraries(example2 gloo)
