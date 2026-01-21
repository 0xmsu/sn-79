file(GLOB filesystem_sources CONFIGURE_DEPENDS "*.cpp")
add_library(filesystem STATIC ${filesystem_sources})
add_library(taosim::filesystem ALIAS filesystem)

target_include_directories(filesystem PUBLIC include)