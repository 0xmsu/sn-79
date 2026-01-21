add_library(replay STATIC)
add_library(taosim::replay ALIAS replay)

file(GLOB replay_sources CONFIGURE_DEPENDS "*.cpp")

target_sources(replay
    PRIVATE
        FILE_SET HEADERS
        BASE_DIRS
            include
    PRIVATE
        ${replay_sources}
)

target_include_directories(replay PUBLIC include)

target_link_libraries(replay
    PRIVATE
        taosim::book
        taosim::decimal
        taosim::filesystem
        taosim::message
        taosim::serialization
        taosim::util
)
