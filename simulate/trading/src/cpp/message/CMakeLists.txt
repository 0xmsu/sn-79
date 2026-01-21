file(GLOB message_sources CONFIGURE_DEPENDS "*.cpp")
add_library(message STATIC ${message_sources})
add_library(taosim::message ALIAS message)

target_include_directories(message
    PUBLIC
        include
        "${SRC_DIR}/base"
)

target_link_libraries(message
    PRIVATE
        taosim::accounting
        taosim::book
        taosim::dsa
        taosim::serialization
        taosim::util
)
