add_library(net STATIC
    net.cpp
    server.cpp
)
add_library(taosim::net ALIAS net)

target_include_directories(net
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SRC_DIR}/serialization"
        "${SRC_DIR}/util"
)

target_link_libraries(net
    PRIVATE
        taosim::decimal
        Boost::boost
)
