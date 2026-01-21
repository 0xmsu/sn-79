file(GLOB logging_sources CONFIGURE_DEPENDS "*.cpp")
add_library(logging STATIC ${logging_sources})
add_library(taosim::logging ALIAS logging)

target_include_directories(logging PUBLIC include)

target_link_libraries(logging
    PUBLIC
        taosim::simulation
)