add_library(dsa INTERFACE)
add_library(taosim::dsa ALIAS dsa)

target_include_directories(dsa INTERFACE include)