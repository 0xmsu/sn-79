file(GLOB checkpoint_sources CONFIGURE_DEPENDS "*.cpp")
add_library(checkpoint STATIC ${checkpoint_sources})
add_library(taosim::checkpoint ALIAS checkpoint)

target_include_directories(checkpoint PUBLIC include)

target_link_libraries(checkpoint
    PRIVATE
        taosim::accounting
        taosim::agent
        taosim::book
        taosim::simulation
        fmt::fmt
)