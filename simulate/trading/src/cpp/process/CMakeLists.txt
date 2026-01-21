file(GLOB process_sources CONFIGURE_DEPENDS "*.cpp")
add_library(process STATIC ${process_sources})
add_library(taosim::process ALIAS process)

target_include_directories(process
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SRC_DIR}"
        "${SRC_DIR}/accounting"
        "${SRC_DIR}/agent"
        "${SRC_DIR}/base"
        "${SRC_DIR}/book"
        "${SRC_DIR}/decimal"
        "${SRC_DIR}/exchange"
        "${SRC_DIR}/mp"
        "${SRC_DIR}/net"
        "${SRC_DIR}/serialization"
        "${SRC_DIR}/util"
)

target_link_libraries(process
    PRIVATE
        taosim::exchange
        taosim::logging
        taosim::message
        taosim::replay
        taosim::simulation
)
